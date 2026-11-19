"""Supabase Postgres/pgvector persistence and permission-filtered retrieval."""
from __future__ import annotations

from typing import Any

import numpy as np
import requests

from .index import Hit
from .ingest import Chunk


class SupabaseError(RuntimeError):
    """A Supabase operation failed; callers should not treat it as an empty result."""


class SupabaseVectorStore:
    def __init__(self, url: str, service_role_key: str, embedder, timeout: float = 20):
        if not url or not service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        self.url = url.rstrip("/")
        self.embedder = embedder
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.session.request(
                method, f"{self.url}/rest/v1/{path}", timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SupabaseError(f"Supabase request failed ({method} {path})") from exc
        if not response.content:
            return None
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise SupabaseError(f"Supabase returned invalid JSON ({method} {path})") from exc

    def search(self, query: str, k: int = 4, user_groups: list[str] | None = None,
               tenant_id: str | None = None, pool: int = 30) -> list[Hit]:
        if not tenant_id:
            raise ValueError("tenant_id is required for Supabase search")
        if k < 1 or pool < 1:
            raise ValueError("k and pool must be positive")
        query_vector = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        if query_vector.shape != (384,):
            raise ValueError(f"Supabase index expects 384-dimensional embeddings, got {query_vector.shape}")
        rows = self._request("POST", "rpc/docqa_search_chunks", json={
            "p_tenant_id": tenant_id,
            "p_user_groups": list(user_groups or []),
            "p_query": query,
            "p_query_embedding": query_vector.tolist(),
            "p_match_count": k,
            "p_candidate_count": max(pool, k),
            "p_rrf_k": 60,
        })
        if not isinstance(rows, list):
            raise SupabaseError("Supabase search returned an unexpected response")
        hits: list[Hit] = []
        for row in rows:
            chunk = Chunk(
                chunk_id=str(row["chunk_id"]),
                doc=str(row["document_name"]),
                loc=str(row["location"]),
                text=str(row["content"]),
                acl=list(row["acl_groups"]),
            )
            hits.append(Hit(
                chunk=chunk,
                score=float(row["score"]),
                bm25=float(row["lexical_score"]),
                dense=float(row["semantic_score"]),
            ))
        return hits

    def upsert_document(self, tenant_id: str, name: str, acl_groups: list[str],
                        uploaded_by: str, chunks: list[dict[str, Any]]) -> str:
        if not tenant_id:
            raise ValueError("tenant_id is required for document ingestion")
        result = self._request("POST", "rpc/docqa_ingest_document", json={
            "p_tenant_id": tenant_id,
            "p_name": name,
            "p_acl_groups": acl_groups,
            "p_uploaded_by": uploaded_by,
            "p_chunks": chunks,
        })
        if not isinstance(result, str):
            raise SupabaseError("Supabase ingestion returned no document ID")
        return result

    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        result = self._request("POST", "rpc/docqa_delete_document", json={
            "p_tenant_id": tenant_id,
            "p_document_id": document_id,
        })
        if not isinstance(result, bool):
            raise SupabaseError("Supabase deletion returned an unexpected response")
        return result

    def write_audit(self, record: dict[str, Any]) -> None:
        self._request(
            "POST",
            "docqa_audit_events",
            json=record,
            headers={"Prefer": "return=minimal"},
        )
