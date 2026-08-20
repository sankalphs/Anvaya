"""Reusable parent-level FAISS retriever for the later Voice → STT → RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from hh_goa_rag.io import read_jsonl
from hh_goa_rag.retrieval import l2_normalize


@dataclass(frozen=True)
class RetrievedParent:
    parent_id: str
    score: float
    chunk_id: str
    text: str


class ParentFaissRetriever:
    """Load a persisted FAISS index and return unique parent passages."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict[str, str]],
        *,
        top_k: int = 10,
        oversample: int = 20,
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunk mapping have different lengths")
        self.index = index
        self.chunks = chunks
        self.top_k = top_k
        self.oversample = oversample

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        chunk_path: str | Path,
        *,
        top_k: int = 10,
        oversample: int = 20,
    ) -> ParentFaissRetriever:
        return cls(
            faiss.read_index(str(index_path)),
            list(read_jsonl(chunk_path)),
            top_k=top_k,
            oversample=oversample,
        )

    def retrieve(self, query_embedding: np.ndarray) -> list[RetrievedParent]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = l2_normalize(query)
        search_k = min(self.index.ntotal, self.top_k * self.oversample)
        scores, positions = self.index.search(query, search_k)
        best_by_parent: dict[str, RetrievedParent] = {}
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            parent_id = str(chunk["parent_id"])
            # `faiss_flat_ip2` is the exact IndexFlatL2 baseline. Because the
            # stored/query vectors are unit-normalized, squared L2 distance is
            # equivalent to cosine similarity via cosine = 1 - distance / 2.
            normalized_score = (
                1.0 - float(score) / 2.0
                if self.index.metric_type == faiss.METRIC_L2
                else float(score)
            )
            candidate = RetrievedParent(
                parent_id=parent_id,
                score=normalized_score,
                chunk_id=str(chunk["chunk_id"]),
                text=str(chunk["text"]),
            )
            current = best_by_parent.get(parent_id)
            if current is None or _prefer_chunk(candidate, current):
                best_by_parent[parent_id] = candidate
        return sorted(best_by_parent.values(), key=lambda item: item.score, reverse=True)[
            : self.top_k
        ]


_CHUNK_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _prefer_chunk(candidate: RetrievedParent, current: RetrievedParent) -> bool:
    """Avoid returning a semantically empty trailing chunk for a parent.

    Fixed-size chunking can leave a final fragment such as ``"है।"``.  That
    fragment can win vector similarity while containing none of the evidence
    needed by the answer generator.  Prefer a substantive candidate for the
    same parent, then use similarity as the tie-breaker.
    """

    candidate_substantive = len(_CHUNK_TOKEN_RE.findall(candidate.text)) >= 5
    current_substantive = len(_CHUNK_TOKEN_RE.findall(current.text)) >= 5
    if candidate_substantive != current_substantive:
        return candidate_substantive
    return candidate.score > current.score
