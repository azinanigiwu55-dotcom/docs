"""Parse documents (PDF, DOCX, TXT/MD) into citation-ready chunks.

Every chunk remembers which document it came from, where in that document
(page for PDFs, section number otherwise) and which groups may read it.
That metadata is what makes citations and permission-aware retrieval possible.

Folders are walked recursively. A document is identified by its path relative
to the folder you ingest ("clients/acme/Engagement_Letter.docx"), and that same
relative path is the key to use in acl.json.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
# Hidden files/folders (.git, .DS_Store) and Office lock files (~$report.docx).
_SKIP_PREFIXES = (".", "~$")


@dataclass
class Chunk:
    chunk_id: str
    doc: str            # path relative to the ingested folder, e.g. "clients/Acme.docx"
    loc: str            # "p.2" for PDFs, "§3" for other formats
    text: str
    acl: list[str] = field(default_factory=lambda: ["admin"])

    @property
    def doc_title(self) -> str:
        return Path(self.doc).stem.replace("_", " ")

    @property
    def index_text(self) -> str:
        """Text used for search. The title helps tell near-identical documents apart."""
        return f"{self.doc_title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ parsing
def _read_docx(path: Path) -> str:
    """Paragraphs and table rows in the order they appear in the document."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    parts: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, doc).text
            if text.strip():
                parts.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                seen, cells = set(), []
                for cell in row.cells:
                    if cell._tc in seen:      # merged cells repeat the same cell
                        continue
                    seen.add(cell._tc)
                    cells.append(cell.text.strip())
                if any(cells):
                    parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_units(path: Path) -> list[tuple[str, str]]:
    """Return [(location_label, text)] for a file. Label is '' when unknown."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        return [("", _read_docx(path))]
    if suffix in {".txt", ".md"}:
        return [("", path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ----------------------------------------------------------------- chunking
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long 'sentence' (no punctuation, a table dump...) into pieces <= max_chars."""
    pieces: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars + 1)
        if cut <= 0:                      # no space to break on: hard cut
            cut = max_chars
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Greedy sentence packing with a small overlap between neighbouring chunks.

    No chunk is longer than max_chars: over-long sentences are split, and the
    overlap carried into the next chunk is trimmed so it never pushes it over.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    sentences = [piece
                 for s in _SENTENCE_SPLIT.split(text) if s.strip()
                 for piece in _split_long(s.strip(), max_chars)]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        if current and size + len(sent) + 1 > max_chars:
            chunks.append("\n".join(current))
            # carry the tail of the previous chunk forward as overlap
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            # ...but never so much that the next chunk exceeds max_chars
            while tail and tail_size + len(sent) + 1 > max_chars:
                tail_size -= len(tail.pop(0)) + 1
            current, size = tail, tail_size
        current.append(sent)
        size += len(sent) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ------------------------------------------------------------------- folder
def load_acl(path: str | Path | None) -> dict[str, list[str]]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def _find_documents(root: Path) -> list[tuple[Path, str]]:
    """Every supported file under root as (path, posix path relative to root), sorted."""
    found = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part.startswith(_SKIP_PREFIXES) for part in rel.parts):
            continue
        if p.suffix.lower() in SUPPORTED:
            found.append((p, rel.as_posix()))
    return sorted(found, key=lambda item: item[1])


def _groups_for(rel: str, acl: dict[str, list[str]], name_counts: Counter,
                default_acl: list[str], issues: list[str]) -> list[str]:
    if rel in acl:
        return acl[rel]
    name = rel.rsplit("/", 1)[-1]
    if name in acl:
        if name_counts[name] == 1:        # a bare file name is fine when it is unambiguous
            return acl[name]
        issues.append(f"{rel}: the ACL entry {name!r} is ambiguous (several folders hold a file "
                      f"with that name), so the default access was used. Key the entry by "
                      f"relative path instead.")
    return default_acl


def ingest_folder(
    docs_dir: str | Path,
    acl: dict[str, list[str]] | None = None,
    default_acl: list[str] | None = None,
    max_chars: int = 900,
    overlap: int = 150,
    issues: list[str] | None = None,
) -> list[Chunk]:
    """Chunk every supported file under docs_dir, including sub-folders.

    ACL entries are keyed by path relative to docs_dir ("clients/Acme.docx"). A bare
    file name still works when only one file has that name. Files with no entry get
    `default_acl` (default: deny-by-default). Use ["all"] only for documents
    that every authenticated or public caller may read.

    Problems that should not pass silently (scanned PDFs with no text, ACL entries
    that match no file, ambiguous ACL keys) are appended to `issues` when a list is
    given, and are always logged as warnings.
    """
    root = Path(docs_dir)
    acl = {k.replace("\\", "/"): v for k, v in (acl or {}).items()}
    default_acl = ["admin"] if default_acl is None else default_acl
    problems: list[str] = issues if issues is not None else []
    start = len(problems)

    documents = _find_documents(root)
    name_counts = Counter(rel.rsplit("/", 1)[-1] for _, rel in documents)
    known = {rel for _, rel in documents} | set(name_counts)
    for key in sorted(set(acl) - known):
        problems.append(f"ACL entry {key!r} matches no document in {root}")

    chunks: list[Chunk] = []
    for path, rel in documents:
        groups = _groups_for(rel, acl, name_counts, default_acl, problems)
        units = read_units(path)
        empty = [loc for loc, text in units if not text.strip()]
        if empty and len(empty) == len(units):
            problems.append(f"{rel}: no extractable text (scanned or image-only?); nothing was "
                            f"indexed. Run OCR on it first.")
            continue
        if empty:
            problems.append(f"{rel}: no extractable text on {', '.join(empty)} (scanned pages?)")
        section = 0
        for loc, text in units:
            for piece in chunk_text(text, max_chars, overlap):
                section += 1
                label = loc or f"§{section}"
                chunks.append(Chunk(
                    chunk_id=f"{rel}::{len(chunks)}",
                    doc=rel, loc=label, text=piece, acl=list(groups),
                ))
    for message in problems[start:]:
        log.warning(message)
    return chunks
