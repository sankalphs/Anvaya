# Gold-query generation evaluation

Status: **model selection is pending blinded human review**. No winning model, Top-K, or prompt
has been declared from diagnostic heuristics.

## Frozen experimental inputs

- Queries: 12 answerable development cases using `stt_reference` (gold text), never Sarvam STT
  transcripts.
- Retriever: BAAI/bge-m3 → sentence chunks capped at 128 words → FAISS HNSW
  (`M=32`, `efConstruction=200`, `efSearch=128`).
- Retrieval snapshot: one Top-10 retrieval per case, cached at
  `cache/generation/gold_contexts_top10.jsonl` and reused for every model call.
- Snapshot fingerprint: `f0a95f0d5851378c60157457b2374da47400d47229f79448b5cc4e5c4b082427`.
- Shared generation condition: prompt `generation-v1.1` / `structured_evidence_ids`,
  temperature 0, reasoning disabled, 192-token output cap, streaming enabled.
- Output schema: `ANSWER` plus answer and retrieved parent IDs, or
  `INSUFFICIENT_CONTEXT` plus an empty answer and no IDs.

Only `SARVAM_API_KEY` is configured. Live calls on 2026-08-18 accepted `sarvam-105b` and
`sarvam-105b-conversations`; the API rejected `sarvam-30b` and `sarvam-m` as deprecated. The
requested 3–5-model ablation therefore cannot be performed without adding an unavailable model or
provider. The experiment contains the two callable models and invents none.

## Measured automated operational metrics

| Model | Success | Total latency P50 / P70 / P95 / P100 | TTFT P50 / P70 / P95 / P100 | Mean output tokens | Recovered retries |
|---|---:|---:|---:|---:|---:|
| sarvam-105b | 12/12 | 1420 / 1591 / 32108 / 32232 ms | 260 / 285 / 339 / 365 ms | 92.6 | 2 |
| sarvam-105b-conversations | 12/12 | 5146 / 5254 / 6662 / 6797 ms | 285 / 291 / 316 / 325 ms | 110.8 | 0 |

Failures and final timeouts were 0/12 for both models. Two `sarvam-105b` cases completed on a
second attempt, so retry-inclusive tail latency is intentionally retained rather than discarded.
The measured generation stage does not meet 200 ms: even median TTFT was above 200 ms. This is not
an end-to-end measurement and makes no end-to-end latency claim.

## Diagnostic heuristics (not quality ground truth)

Both models produced schema-valid structured output on 12/12 cases in the corrected protocol. The
following observed rates were 0% for both: missing citations, citations to IDs outside the supplied
Top-10, and answer numbers absent from all retrieved text. The gold relevant parent appeared in the
cached Top-10 for 12/12 cases. These checks detect obvious protocol violations only; they do not
establish correctness, relevance, or faithfulness.

An earlier integration-debug pass caused the models to enumerate too many evidence IDs and truncate
JSON at the 192-token cap. Prompt v1.1 fixed the instruction to a two-sentence answer and 1–3 direct
evidence IDs. The corrected pass terminated normally for all 24 outputs and is the only pass in the
tracked ablation tables.

## Human/blinded quality judgments

`results/generation_blinded_judgments.csv` contains 24 shuffled outputs with model identity removed,
the question, reference answer, required claims, retrieved evidence, generated answer, and cited
IDs. Its three human score columns are intentionally blank.

Use integer scores from 1 to 5:

- Correctness: 1 = wrong, 3 = materially partial, 5 = fully correct against the reference and
  required claims.
- Relevance: 1 = unrelated, 3 = partly direct or unnecessarily distracting, 5 = direct and concise.
- Faithfulness: 1 = material claims unsupported, 3 = mixed support, 5 = every material claim
  supported by the retrieved evidence.

Human quality is therefore **not yet measured**. Automatic diagnostics have not been relabeled as
human scores or used to select a winner.

## Recommendation and gated next experiments

Provisional latency observation only: `sarvam-105b` has the lower median full-response latency and
lower median TTFT, while `sarvam-105b-conversations` has the lower retry-inclusive P95/P100 in this
single 12-case run. This is not a model recommendation because blinded quality is pending.

- Best model: **pending blinded human scores**.
- Best Top-K: **not run until the winning model is selected**.
- Best prompt: **not run until the winning model and Top-K are selected**.
- Measured quality: **pending**.

After the model sheet is scored and unblinded, run Top-K 1/3/5/10 using only the winner and the fixed
structured-evidence prompt. Judge those outputs blind, choose the smallest Top-K that maintains the
winning model's human quality, then run the three prompt variants at that fixed model and Top-K.
`generation_topk_ablation.csv` and `generation_prompt_ablation.csv` are explicitly marked pending so
missing experiments cannot be mistaken for zero-valued results.

## Fixed protocol for subsequent experiments

1. Use development answerable cases and gold `stt_reference` queries only; never read sealed-test
   data and never substitute STT transcripts in this phase.
2. Validate the frozen retriever configuration and context-cache fingerprint.
3. During model ablation, reuse identical cached Top-10 contexts, prompt version, temperature 0,
   disabled reasoning, 192-token limit, request order, timeout, and retry policy.
4. Record every attempt's final structured observation, total wall-clock latency, TTFT, provider
   token usage, finish reason, failure/timeout, citations, and diagnostic flags as JSONL.
5. Aggregate latency using linear-interpolated P50/P70/P95 and observed maximum P100; include retry
   and timeout duration. Failure rate is failed cases divided by attempted cases.
6. Shuffle outputs under deterministic blind IDs. Human reviewers score correctness, relevance, and
   faithfulness from 1–5 without model identity. Keep scores blank until a human enters them.
7. Select a model using human quality first and latency only as a tiebreaker. Do not use schema or
   citation heuristics as a substitute for quality.
8. With only that model, vary Top-K in order 1, 3, 5, 10 while holding the prompt fixed. Select the
   smallest Top-K that maintains human quality.
9. With model and Top-K fixed, vary only the prompt among strict context-only, context-only plus
   explicit refusal, and structured evidence IDs. Select the fastest prompt that maintains quality
   and valid provenance.
10. Preserve raw runs, append the measured aggregates, record the selected configuration, and keep
    the sealed final evaluation untouched.
