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

## Phase 2: embedding-model ablation

```powershell
hh-goa-ablate --config configs/experiment.yaml embedding run
```

Every candidate uses the same development queries, 128-word non-overlapping chunks, normalized
embeddings, exact FAISS inner-product search, parent-level qrels, and top-K settings. E5 receives
its published `query:`/`passage:` prefixes; Jina receives its published retrieval prompts and task
adapters. IndicBERT is loaded through its model-card-prescribed bidirectional causal-LM wrapper and
mean-pooled because the released checkpoint is not a retrieval-tuned sentence encoder.

The current Transformers 5 runtime needs two recorded compatibility adapters. Alibaba's custom GTE
implementation leaves non-persistent position and RoPE buffers uninitialized after meta-device
loading, so they are deterministically reconstructed from the checkpoint config. Jina's pinned
secondary implementation is stored in the project-local dynamic-module cache and receives the
current `post_init` lifecycle call. These adapters do not change learned tensors.

The winner is selected by the predeclared quality priority `nDCG@10`, `MRR@10`, `Recall@10`, with
combined query-embedding and retrieval P50 latency only as the final tie-breaker. Results are saved
to `results/embedding_ablation.csv`; per-run JSON and all embeddings remain in ignored local caches.

## Phase 3: chunking ablation

```powershell
hh-goa-ablate --config configs/experiment.yaml chunking run
```

The winning embedding model is held fixed while fixed-size, overlapping, sentence-based,
semantic, and parent-child strategies are compared on the same development queries and
parent-level qrels. Semantic boundaries use adjacent normalized sentence-embedding similarity
with a predeclared threshold; no qrels are used to tune boundaries. All strategies use normalized
embeddings and exact FAISS inner-product search.

The winner is selected by the same quality-first priority as Phase 2, with retrieval P50 latency
only as a final tie-breaker. The comparison is saved to `results/chunking_ablation.csv` and the
machine-readable selected configuration to `results/chunking_winner.json`.
