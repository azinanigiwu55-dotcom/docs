from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from docqa.api import (
    AuthenticationError,
    Principal,
    Settings,
    SupabaseAuthVerifier,
    create_app,
)
from docqa.index import Hit
from docqa.ingest import Chunk
from docqa.supabase_store import SupabaseVectorStore


class FakeAuth:
    def authenticate(self, token: str) -> Principal:
        principals = {
            "staff": Principal("user-staff", "tenant-a", ("staff",), False),
            "admin": Principal("user-admin", "tenant-a", ("staff",), True),
            "other-tenant": Principal("user-other", "tenant-b", ("staff",), False),
        }
        try:
            return principals[token]
        except KeyError as exc:
            raise AuthenticationError("invalid token") from exc


class FakeEmbedder:
    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return np.ones(384, dtype=np.float32)


class FakeStore:
    def __init__(self):
        self.documents: dict[tuple[str, str], dict] = {}
        self.audit: list[dict] = []
        self.last_search: tuple[str, tuple[str, ...], str] | None = None

    def search(self, query, k=4, user_groups=None, tenant_id=None, pool=30):
        groups = tuple(user_groups or [])
        self.last_search = (tenant_id, groups, query)
        hits = []
        for (tenant, _name), document in self.documents.items():
            if tenant != tenant_id or not (
                "all" in document["groups"] or set(groups) & set(document["groups"])
            ):
                continue
            for i, row in enumerate(document["chunks"][:k]):
                chunk = Chunk(str(i), document["name"], row["location"], row["content"],
                              document["groups"])
                hits.append(Hit(chunk, 1.0, 1.0, 1.0))
        return hits

    def upsert_document(self, tenant_id, name, acl_groups, uploaded_by, chunks):
        document_id = str(uuid4())
        self.documents[(tenant_id, name)] = {
            "id": document_id, "name": name, "groups": acl_groups, "chunks": chunks,
        }
        return document_id

    def delete_document(self, tenant_id, document_id):
        for key, document in list(self.documents.items()):
            if key[0] == tenant_id and document["id"] == document_id:
                del self.documents[key]
                return True
        return False

    def write_audit(self, record):
        self.audit.append(record)


class SimpleAnswer:
    def generate(self, question, hits, messages):
        return f"The retention period is 7 years [1]."


@pytest.fixture
def api():
    store = FakeStore()
    app = create_app(
        settings=Settings(),
        auth_verifier=FakeAuth(),
        store=store,
        embedder=FakeEmbedder(),
        backend=SimpleAnswer(),
    )
    return TestClient(app), store


def test_query_requires_authenticated_identity(api):
    client, _store = api
    response = client.post("/v1/query", json={"question": "How long is retention?"})
    assert response.status_code == 401


def test_query_rejects_blank_questions(api):
    client, _store = api
    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer staff"},
        json={"question": "   "},
    )
    assert response.status_code == 422


def test_query_uses_verified_tenant_and_groups_not_request_claims(api):
    client, store = api
    store.upsert_document("tenant-a", "policy.txt", ["staff"], "user-admin", [{
        "location": "§1",
        "content": "The retention period is 7 years.",
        "embedding": [1.0] * 384,
    }])
    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer staff"},
        json={"question": "How long is the retention period?", "tenant_id": "tenant-b",
              "user_groups": ["admin"]},
    )
    assert response.status_code == 422

    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer staff"},
        json={"question": "How long is the retention period?"},
    )
    assert response.status_code == 200
    assert store.last_search == ("tenant-a", ("staff",), "How long is the retention period?")
    assert store.audit[-1]["user_id"] == "user-staff"


def test_query_cannot_retrieve_across_tenants(api):
    client, store = api
    store.upsert_document("tenant-b", "policy.txt", ["staff"], "user-other", [{
        "location": "§1",
        "content": "The retention period is 7 years.",
        "embedding": [1.0] * 384,
    }])
    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer staff"},
        json={"question": "How long is the retention period?"},
    )
    assert response.status_code == 200
    assert response.json()["found"] is False


def test_ingestion_is_admin_only_and_requires_acl(api):
    client, store = api
    response = client.post(
        "/v1/documents",
        headers={"Authorization": "Bearer staff"},
        data={"acl_groups": "staff"},
        files={"file": ("policy.txt", b"Keep records for 7 years.", "text/plain")},
    )
    assert response.status_code == 403
    assert not store.documents

    response = client.post(
        "/v1/documents",
        headers={"Authorization": "Bearer admin"},
        data={"acl_groups": "staff"},
        files={"file": ("policy.txt", b"Keep records for 7 years.", "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["chunks"] == 1
    assert ("tenant-a", "policy.txt") in store.documents

    response = client.post(
        "/v1/documents",
        headers={"Authorization": "Bearer admin"},
        data={"acl_groups": "staff"},
        files={"file": ("broken.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 422


def test_document_delete_is_tenant_scoped_and_admin_only(api):
    client, store = api
    document_id = store.upsert_document("tenant-b", "foreign.txt", ["staff"], "user-other", [])
    response = client.delete(
        f"/v1/documents/{document_id}",
        headers={"Authorization": "Bearer admin"},
    )
    assert response.status_code == 200
    assert response.json()["deleted"] is False
    assert ("tenant-b", "foreign.txt") in store.documents


def test_supabase_search_requires_tenant_and_passes_groups_to_filtered_rpc():
    store = SupabaseVectorStore("https://example.supabase.co", "service-key", FakeEmbedder())
    calls = []

    class Response:
        content = b"[]"

        def raise_for_status(self):
            pass

        def json(self):
            return [{
                "chunk_id": "chunk-1",
                "document_name": "handbook.txt",
                "location": "§2",
                "content": "Employees receive 20 days of leave.",
                "acl_groups": ["staff"],
                "score": 0.04,
                "lexical_score": 0.2,
                "semantic_score": 0.8,
            }]

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Response()

    store.session.request = request
    with pytest.raises(ValueError, match="tenant_id"):
        store.search("leave", user_groups=["staff"])
    hits = store.search("leave", user_groups=["staff"], tenant_id="tenant-a")
    assert hits[0].chunk.doc == "handbook.txt"
    assert calls[0][2]["json"]["p_tenant_id"] == "tenant-a"
    assert calls[0][2]["json"]["p_user_groups"] == ["staff"]


def test_supabase_auth_trusts_app_metadata_not_user_metadata(monkeypatch):
    import docqa.api as api_module

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "id": "user-1",
                "app_metadata": {
                    "tenant_id": "tenant-a",
                    "groups": ["staff", "all"],
                    "roles": [],
                },
                "user_metadata": {
                    "tenant_id": "tenant-b",
                    "groups": ["admin"],
                    "roles": ["admin"],
                },
            }

    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return Response()

    monkeypatch.setattr(api_module.requests, "get", get)
    verifier = SupabaseAuthVerifier("https://example.supabase.co", "anon-key")
    principal = verifier.authenticate("token")
    assert principal == Principal("user-1", "tenant-a", ("staff",), False)
    assert seen["headers"]["apikey"] == "anon-key"
