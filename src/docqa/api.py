"""Controlled HTTP API backed by Supabase Auth and Postgres/pgvector."""
from __future__ import annotations

import os
import tempfile
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID

import requests
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator
from fastapi.responses import JSONResponse
from docx.opc.exceptions import PackageNotFoundError
from pypdf.errors import PdfReadError

from .answer import ExtractiveBackend, HFBackend, OllamaBackend, answer
from .index import SentenceTransformerEmbedder
from .ingest import SUPPORTED, ingest_folder
from .profile import Profile
from .supabase_store import SupabaseError, SupabaseVectorStore

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_INGEST_CHUNKS = 2000
MAX_ACL_GROUPS = 50


@dataclass(frozen=True)
class Settings:
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    backend: str = "extractive"
    model: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv

        load_dotenv()
        return cls(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            embedding_model=os.getenv("DOCQA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            backend=os.getenv("DOCQA_BACKEND", "extractive").lower(),
            model=os.getenv("DOCQA_MODEL"),
        )


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    groups: tuple[str, ...]
    is_admin: bool


class AuthenticationError(RuntimeError):
    pass


class IdentityProviderUnavailable(RuntimeError):
    pass


class SupabaseAuthVerifier:
    """Validate access tokens through Supabase Auth and trust app_metadata only."""

    def __init__(self, url: str, anon_key: str, timeout: float = 10):
        if not url or not anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY are required")
        self.url = url.rstrip("/")
        self.anon_key = anon_key
        self.timeout = timeout

    def authenticate(self, token: str) -> Principal:
        try:
            response = requests.get(
                f"{self.url}/auth/v1/user",
                headers={"apikey": self.anon_key, "Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise IdentityProviderUnavailable("Supabase Auth could not be reached") from exc
        if response.status_code in (401, 403):
            raise AuthenticationError("Access token is invalid or expired")
        if response.status_code >= 500:
            raise IdentityProviderUnavailable("Supabase Auth is unavailable")
        try:
            response.raise_for_status()
            user = response.json()
        except (requests.RequestException, requests.JSONDecodeError) as exc:
            raise IdentityProviderUnavailable("Supabase Auth returned an invalid response") from exc

        metadata = user.get("app_metadata")
        if not isinstance(metadata, dict):
            raise AuthenticationError("Trusted application metadata is missing")
        tenant_id = metadata.get("tenant_id")
        groups = metadata.get("groups", [])
        roles = metadata.get("roles", [])
        user_id = user.get("id")
        if (not isinstance(tenant_id, str) or not tenant_id.strip()
                or not isinstance(groups, list) or not all(isinstance(g, str) for g in groups)
                or not isinstance(roles, list) or not all(isinstance(r, str) for r in roles)
                or not isinstance(user_id, str)):
            raise AuthenticationError("Trusted tenant, group, or user claims are invalid")
        trusted_groups = {group for group in groups if group and group != "all"}
        if "admin" in roles:
            trusted_groups.add("admin")
        return Principal(
            user_id=user_id,
            tenant_id=tenant_id,
            groups=tuple(sorted(trusted_groups)),
            is_admin="admin" in roles,
        )


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class CitationResponse(BaseModel):
    document: str
    location: str
    quote: str


class QueryResponse(BaseModel):
    answer: str
    found: bool
    reason: str
    citations: list[CitationResponse]


class IngestResponse(BaseModel):
    document_id: str
    name: str
    chunks: int


class DeleteResponse(BaseModel):
    deleted: bool


def _backend(settings: Settings):
    if settings.backend == "extractive":
        return ExtractiveBackend()
    if settings.backend == "ollama":
        return OllamaBackend(model=settings.model or "qwen2.5:3b")
    if settings.backend == "hf":
        return HFBackend(model_id=settings.model or "Qwen/Qwen2.5-3B-Instruct")
    raise ValueError("DOCQA_BACKEND must be extractive, ollama, or hf")


def _principal(request: Request, authorization: Annotated[str | None, Header()] = None) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A Supabase Bearer access token is required")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="A Supabase Bearer access token is required")
    try:
        return request.app.state.auth_verifier.authenticate(token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except IdentityProviderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _admin(principal: Annotated[Principal, Depends(_principal)]) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Tenant administrator role is required")
    return principal


def create_app(settings: Settings | None = None, auth_verifier=None, store=None,
               embedder=None, backend=None) -> FastAPI:
    config = settings or Settings.from_env()

    def initialize_services(application: FastAPI) -> None:
        if application.state.auth_verifier is None:
            application.state.auth_verifier = SupabaseAuthVerifier(
                config.supabase_url, config.supabase_anon_key
            )
        if application.state.embedder is None:
            application.state.embedder = SentenceTransformerEmbedder(config.embedding_model)
        if application.state.store is None:
            application.state.store = SupabaseVectorStore(
                config.supabase_url, config.supabase_service_role_key, application.state.embedder
            )
        if application.state.backend is None:
            application.state.backend = _backend(config)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        initialize_services(application)
        yield

    app = FastAPI(title="Private Document Q&A API", version="0.2.0", lifespan=lifespan)
    app.state.settings = config
    app.state.auth_verifier = auth_verifier
    app.state.store = store
    app.state.embedder = embedder
    app.state.backend = backend

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.post("/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request,
              principal: Annotated[Principal, Depends(_principal)]):
        profile = Profile(tenant_id=principal.tenant_id)
        result = answer(
            payload.question,
            request.app.state.store,
            request.app.state.backend,
            k=payload.k,
            user_groups=list(principal.groups),
            user=principal.user_id,
            profile=profile,
            tenant_id=principal.tenant_id,
        )
        request.app.state.store.write_audit({
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "action": "query",
            "question": payload.question,
            "documents_retrieved": sorted({hit.chunk.doc for hit in result.hits}),
            "documents_cited": sorted({citation["doc"] for citation in result.citations}),
            "outcome": result.reason,
        })
        return QueryResponse(
            answer=result.text,
            found=result.found,
            reason=result.reason,
            citations=[
                CitationResponse(
                    document=citation["doc"],
                    location=citation["loc"],
                    quote=citation.get("quote", ""),
                )
                for citation in result.citations
            ],
        )

    @app.post("/v1/documents", response_model=IngestResponse, status_code=201)
    def ingest_document(
        request: Request,
        principal: Annotated[Principal, Depends(_admin)],
        file: Annotated[UploadFile, File()],
        acl_groups: Annotated[str, Form()],
    ):
        name = Path((file.filename or "").replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if (not name or name in {".", ".."} or len(name) > 255
                or any(ord(character) < 32 for character in name) or suffix not in SUPPORTED):
            raise HTTPException(status_code=415, detail="Upload a PDF, DOCX, TXT, or Markdown file")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 20 MiB upload limit")
        groups = sorted({group.strip() for group in acl_groups.split(",") if group.strip()})
        if (not groups or len(groups) > MAX_ACL_GROUPS
                or any(len(group) > 100 for group in groups)):
            raise HTTPException(
                status_code=422,
                detail=f"Provide between 1 and {MAX_ACL_GROUPS} ACL groups, each at most 100 characters",
            )

        with tempfile.TemporaryDirectory(prefix="docqa-ingest-") as directory:
            path = Path(directory) / name
            path.write_bytes(content)
            issues: list[str] = []
            try:
                chunks = ingest_folder(
                    directory,
                    acl={name: groups},
                    default_acl=["admin"],
                    issues=issues,
                )
            except (OSError, ValueError, zipfile.BadZipFile, PackageNotFoundError, PdfReadError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not parse the uploaded document ({type(exc).__name__})",
                ) from exc
        if issues:
            raise HTTPException(status_code=422, detail={"ingestion_issues": issues})
        if not chunks:
            raise HTTPException(status_code=422, detail="The file contains no extractable text")
        if len(chunks) > MAX_INGEST_CHUNKS:
            raise HTTPException(status_code=413, detail="Document exceeds the 2,000 chunk ingestion limit")

        embeddings = request.app.state.embedder.encode_docs([chunk.index_text for chunk in chunks])
        if embeddings.shape != (len(chunks), 384):
            raise HTTPException(status_code=500, detail="Configured embedder does not produce 384 dimensions")
        rows = [
            {
                "location": chunk.loc,
                "content": chunk.text,
                "embedding": [round(float(value), 6) for value in embeddings[i]],
            }
            for i, chunk in enumerate(chunks)
        ]
        document_id = request.app.state.store.upsert_document(
            principal.tenant_id, name, groups, principal.user_id, rows
        )
        request.app.state.store.write_audit({
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "action": "ingest",
            "document_name": name,
            "outcome": "ok",
        })
        return IngestResponse(document_id=document_id, name=name, chunks=len(chunks))

    @app.delete("/v1/documents/{document_id}", response_model=DeleteResponse)
    def delete_document(document_id: str, request: Request,
                        principal: Annotated[Principal, Depends(_admin)]):
        try:
            canonical_id = str(UUID(document_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="document_id must be a UUID") from exc
        deleted = request.app.state.store.delete_document(principal.tenant_id, canonical_id)
        request.app.state.store.write_audit({
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "action": "delete",
            "outcome": "ok" if deleted else "not_found",
        })
        return DeleteResponse(deleted=deleted)

    @app.exception_handler(SupabaseError)
    async def supabase_error_handler(_request: Request, exc: SupabaseError):
        return JSONResponse(
            status_code=503,
            content={"detail": "The document service is temporarily unavailable"},
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("docqa.api:app", host="127.0.0.1", port=8000)
