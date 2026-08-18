"""Reusable parent-level FAISS retriever for the later Voice → STT → RAG pipeline."""

from __future__ import annotations

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
        result: list[RetrievedParent] = []
        seen: set[str] = set()
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            parent_id = str(chunk["parent_id"])
            if parent_id in seen:
                continue
            seen.add(parent_id)
            result.append(
                RetrievedParent(
                    parent_id=parent_id,
                    score=float(score),
                    chunk_id=str(chunk["chunk_id"]),
                    text=str(chunk["text"]),
                )
            )
            if len(result) == self.top_k:
                break
        return result
