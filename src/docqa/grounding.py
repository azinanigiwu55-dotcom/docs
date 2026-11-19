"""Cheap, model-free checks that an answer really comes from its cited sources.

A model can cite [1] and still state a figure that is not in source [1]. For
contracts, invoices and policies a wrong number is the costly failure, so every
figure in an answer must appear in the sources it cites. This does not prove an
answer is right, but it catches invented and hijacked numbers for free.
"""
from __future__ import annotations

import re

from .index import tokenize

_CITE = re.compile(r"\[\d+\]")
_CITE_NUMBER = re.compile(r"\[(\d+)\]")
_CLAIM_SENTENCE = re.compile(r"(?<=[.!?])\s+(?!\[\d+\])|\n+")
# "48,500", "48 500" (space or non-breaking space) and "48500" count as the same figure.
# The group before the first separator has 1-3 digits, so "2024 100" stays two figures.
_SEP = "[ ,\u00a0\u202f]"
_THOUSANDS = re.compile(rf"(?<!\d)\d{{1,3}}(?:{_SEP}\d{{3}})+(?!\d)")
_SEPARATOR = re.compile(_SEP)
_FIGURE = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {"the", "a", "an", "of", "for", "to", "in", "is", "are", "was", "were", "by", "and",
         "or", "with", "be", "at", "this", "that", "it", "its", "on", "as", "per"}
_CLAIM_TERMS = _STOP | {"according", "document", "documents", "source", "sources", "states",
                        "says", "shows", "indicates", "reported", "provided"}


def merge_thousands(text: str) -> str:
    """Remove thousands separators inside numbers: 'R48 500' and 'R48,500' both become 'R48500'."""
    return _THOUSANDS.sub(lambda m: _SEPARATOR.sub("", m.group(0)), text)


def figures(text: str) -> set[str]:
    """Numbers in a text, normalised (thousands separators and citation marks removed)."""
    return set(_FIGURE.findall(merge_thousands(_CITE.sub("", text))))


def ungrounded_figures(answer_text: str, question: str, sources: list[str]) -> list[str]:
    """Figures in the answer that do not occur in its cited source text."""
    del question
    allowed: set[str] = set()
    for s in sources:
        allowed |= figures(s)
    return sorted(figures(answer_text) - allowed)


def unsupported_claims(answer_text: str, sources_by_number: dict[int, str],
                       min_coverage: float = 0.6) -> list[str]:
    """Find uncited or lexically unsupported answer sentences.

    This deterministic check is a conservative guard, not a proof of entailment.
    Each answer sentence must cite at least one supplied source, and at least 60%
    of its non-trivial terms must occur in the cited source text.
    """
    failures: list[str] = []
    for sentence in (s.strip() for s in _CLAIM_SENTENCE.split(answer_text) if s.strip()):
        numbers = {int(n) for n in _CITE_NUMBER.findall(sentence) if int(n) in sources_by_number}
        claim_terms = {t for t in tokenize(_CITE.sub("", sentence)) if t not in _CLAIM_TERMS}
        if not claim_terms:
            continue
        if not numbers:
            failures.append(sentence)
            continue
        source_terms = {
            t for number in numbers for t in tokenize(sources_by_number[number])
            if t not in _CLAIM_TERMS
        }
        coverage = len(claim_terms & source_terms) / len(claim_terms)
        if coverage < min_coverage:
            failures.append(sentence)
    return failures


def best_quote(answer_text: str, chunk_text: str, limit: int = 240) -> str:
    """The sentence of a source that best matches the answer, for showing next to a citation."""
    wanted = {t for t in tokenize(_CITE.sub("", answer_text))} - _STOP
    sentences = [s.strip() for s in _SENTENCE.split(chunk_text) if s.strip()]
    if not sentences:
        return ""
    best = max(sentences, key=lambda s: len(wanted & set(tokenize(s))))
    return best if len(best) <= limit else best[: limit - 1].rstrip() + "…"
