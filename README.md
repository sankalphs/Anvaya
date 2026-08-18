# HH Goa retrieval ablation

Experimental retrieval stage for the HH Goa Voice-RAG project. This repository selects an
embedding model, chunking strategy, and vector index using a fixed, leakage-safe evaluation
protocol on [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).
It intentionally contains no LLM generation, STT, frontend, or final RAG application.

## Environment

Python 3.11–3.14 and an NVIDIA GPU are recommended. Install the project and experiment extras:

```powershell
python -m pip install -e ".[experiment,dev]"
```

All generated data, embeddings, indexes, and model downloads are scoped beneath this repository's
`data/` and `cache/` directories and are ignored by Git.

## Phase 1: prepare the dataset

```powershell
hh-goa-ablate --config configs/experiment.yaml dataset prepare
```

The loader discovers the repository's real parquet inventory, resolves `language: auto` to the
dataset's Hindi default, and pins the resolved Hub commit SHA. Recent `datasets` releases do not
recognize the repository's legacy per-language loading script, and its parquet files each contain
one very large row group. The loader therefore downloads one pinned file at a time, scans it in
Arrow batches, caches compact JSONL artifacts, and optionally removes only the transient raw file.

The upstream training split supplies development queries for all ablations. The upstream
validation split is sealed as test and is evaluated only once on the final winning stack. Each
split contains a global pooled corpus plus queries and qrels. Qrels identify parent passages, so
later chunking strategies are compared against exactly the same gold unit.

Run checks with:

```powershell
python -m pytest
ruff check .
```
