"""Deterministic passage chunking with stable parent/child identities."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from hh_goa_rag.config import stable_fingerprint

WORD_PATTERN = re.compile(r"\S+", re.UNICODE)


def fixed_word_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be non-negative and smaller than size")
    words = WORD_PATTERN.findall(text)
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def chunk_corpus(
    corpus: Iterable[dict[str, Any]], strategy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply a configured chunker and retain stable parent passage IDs."""
    name = strategy["strategy"]
    if name != "fixed_words":
        raise ValueError(f"Unsupported chunking strategy: {name}")
    chunks: list[dict[str, Any]] = []
    for parent in corpus:
        texts = fixed_word_chunks(
            parent["text"], int(strategy["size"]), int(strategy.get("overlap", 0))
        )
        for position, text in enumerate(texts):
            identity = {
                "parent_id": parent["passage_id"],
                "position": position,
                "strategy": strategy,
                "text": text,
            }
            chunks.append(
                {
                    "chunk_id": f"c-{stable_fingerprint(identity, 24)}",
                    "parent_id": parent["passage_id"],
                    "position": position,
                    "text": text,
                }
            )
    return chunks

