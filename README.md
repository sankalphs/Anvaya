# HH Goa Voice-RAG evaluation

Evaluation and integration repository for the HH Goa Voice-RAG project. The complete selected stack
is frozen as Sarvam Saaras v3 STT → BGE-M3 → sentence chunks capped at 128 words → FAISS HNSW
(`M=32`, `efConstruction=200`, `efSearch=128`) → deterministic guardrails → `sarvam-105b` →
Top-10 → `strict_context_only` → deterministic grounding validation.

The repository contains retrieval, STT, generation, routing/guardrail, and complete-pipeline
evaluation harnesses. It does not include a frontend, TTS, deployment, or production service.

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

All vectors receive a final float32 L2 normalization at the shared retrieval boundary. This
corrects the small unit-norm drift introduced when a model performs its internal normalization in
bfloat16, and ensures inner product is cosine-equivalent for every experiment and backend.

## Phase 4: index and local-storage ablation

```powershell
hh-goa-ablate --config configs/experiment.yaml index run
```

This stage reuses the exact winning corpus and query vectors; it does not re-embed text. It compares
FAISS FlatIP, HNSW (`M=32`, `efConstruction=200`, `efSearch=128`), IVF-Flat (`nlist=128`,
`nprobe=16`), exact Qdrant embedded/local cosine search, and Chroma local HNSW cosine search
(`M=32`, `efConstruction=200`, `efSearch=128`, eight threads). Each backend is warmed up before
timing 1,000 individual queries, and reports P50/P70/P95/P100 latency, indexing time, process RAM
delta, estimated resident index size, and persistent disk size.

The selection remains quality-first (`nDCG@10`, `MRR@10`, `Recall@10`), followed only on ties by
P95 latency, P50 latency, and disk size. Results are saved to `results/index_ablation.csv`; the
selected backend and exact configuration are saved to `results/index_winner.json`.

## Phase 5: sealed test and retriever handoff

```powershell
hh-goa-ablate --config configs/experiment.yaml finalize run
```

This command evaluates only the already-selected stack on the sealed test split and refuses to
silently re-evaluate a different configuration once `results/final_test.json` exists. It writes the
final recommendation and a machine-readable retriever configuration. It then removes losing model
directories only when they are direct children of `cache/models`, belong to an embedding candidate,
and contain this project's exact `.hh_goa_model.json` ownership marker. The winning model and every
cache outside that directory are preserved.

The reusable `hh_goa_rag.retriever.ParentFaissRetriever` loads the persisted index and chunk mapping
and accepts query embeddings. The later STT evaluation uses that boundary without changing the
selected model, chunking, index, or retrieval parameters.

## Phase 6: real Sarvam STT evaluation

Install the STT and development extras:

```powershell
python -m pip install -e ".[experiment,stt,dev]"
```

Copy `.env.example` to the ignored `.env` file and set `SARVAM_API_KEY`. The provider and
configuration are fixed to Sarvam AI `saaras:v3` with `mode="transcribe"`; no alternate STT
provider is implemented. List microphone devices and record the pending real-human samples with:

```powershell
python eval/record_audio.py --list-devices
python eval/record_audio.py --all-pending --speaker-id speaker-01 --device 5
```

The recorder writes 16 kHz mono PCM WAV files under `eval/audio` and atomically marks manifest rows
ready. Pending or missing audio is never scored. Once recordings are ready, run both Sarvam
integration modes and the frozen-retriever impact evaluation:

```powershell
python eval/evaluate_stt.py --manifest eval/stt_manifest.jsonl --run-id sarvam-real-001 --device auto
```

This produces per-sample STT results, gold-text versus transcript retrieval degradation, raw run
observations, and a data-driven REST-versus-streaming recommendation. The sealed test remains
untouched.

## Phase 7: generation selection

The generation evaluation uses a single cached gold-query Top-10 retrieval snapshot so model,
Top-K, and prompt comparisons do not alter STT or retrieval. The selected configuration is
`sarvam-105b` → Top-10 → `strict_context_only`. Its measured generation-only latency was
P50/P70/P95/P100 1580/1966/7009/11849 ms, so this stack cannot satisfy a complete-pipeline target
below 200 ms. See `results/generation_recommendation.md` for the blinded qualitative review and
ablation details.

## Phase 8: deterministic guardrails

`hh_goa_rag.guardrails` routes structured outcomes among `ANSWER`, `INSUFFICIENT_CONTEXT`,
`OFF_TOPIC`, `UNSAFE`, `STT_FAILURE`, and `SYSTEM_ERROR`. Input policy checks run before retrieval;
the frozen evidence rule uses Top-1 ≥ 0.67 or the fixed Top-K consistency rescue; generation output
must pass schema and retrieved-citation validation. These rules add no LLM judge call.

The 24-case development routing set scored 24/24 with no false answers or false refusals and
guardrail-only P50/P70/P95/P100 latency of approximately 0.020/0.022/0.034/0.069 ms. These are
development-only threshold-selection results, not real-voice E2E measurements. See
`results/guardrail_recommendation.md`.

## Phase 9: complete Voice-RAG integration

The `hh_goa_rag.harness.VoiceRAGHarness` implements:

```text
audio
→ Sarvam STT
→ input and policy guardrails
→ BGE-M3 query embedding
→ FAISS Top-10 vector search
→ evidence guardrail
→ sarvam-105b generation
→ grounding validation
→ structured response with provenance and stage timings
```

Run the complete evaluator with:

```powershell
python eval/evaluate_e2e.py
```

The 24-case recording manifest currently has 0/24 real recordings, so formal real-voice completion,
quality, WER, retrieval-degradation, latency-budget, category, and failure metrics are explicitly
pending. `results/e2e_*.csv` contains blank pending rows rather than synthetic values. Text-path
integration and ten failure-mode checks are recorded separately as smoke tests and never enter the
formal latency tables. See `results/e2e_recommendation.md`.
