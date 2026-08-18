# HH Goa retrieval-stack recommendation

## Selected stack

**Best embedding model → Best chunking strategy → Best index/storage system**

**`BAAI/bge-m3` → sentence-based packing (`max_words=128`) → FAISS HNSW
(`M=32`, `efConstruction=200`, `efSearch=128`, float32 normalized inner product)**

Selection used only the development split. The upstream validation-derived test split remained
sealed until all three winners and their configurations were fixed.

## Exact reproducibility configuration

- Dataset: `ai4bharat/MSMARCO-XI`, auto-resolved language `hi`, Hub revision `bf5cdc1f26e581e519018e434db14edd1b77602b`
- Processed dataset artifact: `data\processed\23828c1c95c62c20`
- Random seed: `20260818`
- Embedding checkpoint: `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`
- Embedding dimension/dtype: 1024 / inference `bfloat16`, persisted `float32`
- Normalization: final float32 L2 (`float32_l2_v1`); similarity is cosine-equivalent inner product
- Chunking: punctuation-aware sentence packing, maximum 128 whitespace-delimited words,
  parent-level qrels
- Index: FAISS `IndexHNSWFlat`, `M=32`, `efConstruction=200`, `efSearch=128`, `METRIC_INNER_PRODUCT`
- Retrieval: top 10 unique parents, search oversampling 20×, 20 warm-up queries
- Evaluation: 1,000 fixed dev queries per ablation; 1,000 sealed test queries exactly once

## Embedding ablation (development)

| model | recall_at_10 | mrr_at_10 | ndcg_at_10 | query_embedding_p50_ms |
| --- | --- | --- | --- | --- |
| BAAI/bge-m3 | 0.8578 | 0.5039 | 0.5873 | 9.1646 |
| intfloat/multilingual-e5-base | 0.8421 | 0.5062 | 0.5857 | 4.9593 |
| Alibaba-NLP/gte-multilingual-base | 0.8217 | 0.4726 | 0.5544 | 5.4121 |
| jinaai/jina-embeddings-v3 | 0.2953 | 0.1331 | 0.1699 | 55.9051 |
| ai4bharat/IndicBERT-v3-4B | 0.3438 | 0.1624 | 0.2040 | 34.6878 |

BGE-M3 won the predeclared lexicographic quality priority: nDCG@10, then MRR@10, then Recall@10;
latency was used only after quality ties.

## Chunking ablation (development)

| strategy | recall_at_10 | mrr_at_10 | ndcg_at_10 | retrieval_p50_ms |
| --- | --- | --- | --- | --- |
| fixed_size | 0.8578 | 0.5039 | 0.5873 | 1.5932 |
| overlapping | 0.8524 | 0.4956 | 0.5794 | 1.7624 |
| sentence_based | 0.8625 | 0.5073 | 0.5908 | 1.5964 |
| semantic | 0.8469 | 0.4978 | 0.5795 | 2.3307 |
| parent_child | 0.8296 | 0.4807 | 0.5624 | 4.5994 |

Sentence packing provided the highest nDCG@10 and Recall@10 while keeping corpus growth close to
fixed-size chunking.

## Index/storage ablation (development)

| backend | recall_at_10 | mrr_at_10 | ndcg_at_10 | retrieval_p50_ms | retrieval_p95_ms | indexing_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| faiss_flat_ip | 0.8625 | 0.5073 | 0.5908 | 1.5890 | 1.8279 | 31.0690 |
| faiss_hnsw | 0.8625 | 0.5073 | 0.5908 | 0.3425 | 0.4395 | 303.8167 |
| faiss_ivf_flat | 0.8387 | 0.5010 | 0.5803 | 0.1752 | 0.2445 | 89.0499 |
| qdrant_local | 0.8625 | 0.5073 | 0.5908 | 40.5374 | 53.7085 | 19799.6287 |
| chroma_local | 0.8625 | 0.5073 | 0.5908 | 2.6992 | 3.3179 | 1498.1449 |

FAISS HNSW matched the best quality metrics and won the latency tie-breaker. IVF-Flat was faster but
reduced Recall@10. Qdrant embedded/local was exact but is a brute-force implementation; Chroma local
matched exact quality but had higher retrieval latency and resource use than FAISS HNSW.

## Sealed test result

| recall_at_1 | recall_at_3 | recall_at_5 | recall_at_10 | mrr_at_10 | ndcg_at_10 | retrieval_p50_ms | retrieval_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3464 | 0.6495 | 0.7899 | 0.8908 | 0.5371 | 0.6212 | 0.3876 | 0.5309 |

- Test chunks: 10228
- Indexing time: 315.4044 ms
- Index disk size: 42.60 MiB
- Query embedding P50: 8.5039 ms
- Final index artifact: `cache\indexes\final\3b19f7581e6e195f\retriever.faiss`
- Chunk-to-parent artifact: `data\processed\23828c1c95c62c20\chunks\final-test-3b19f7581e6e195f.jsonl`

`hh_goa_rag.retriever.ParentFaissRetriever` loads these two artifacts and accepts an already encoded
query vector, so it can be placed directly after the future STT/query-embedding boundary without
adding generation, speech, or frontend code here.

## Project-local model cleanup

The winning BGE-M3 directory was preserved. Only direct children of `cache\models` carrying the
exact `.hh_goa_model.json` ownership marker for this experiment were eligible for deletion. Removed:

- `ai4bharat/IndicBERT-v3-4B` (7.26 GiB)
- `Alibaba-NLP/gte-multilingual-base` (0.58 GiB)
- `intfloat/multilingual-e5-base` (3.15 GiB)
- `jinaai/jina-embeddings-v3` (2.15 GiB)

Freed 13.14 GiB. No global Hugging Face cache, model outside
this repository, processed dataset, embeddings, index, or winning model was deleted.

## Scope and caveats

MSMARCO-XI has no Konkani configuration, so the repository default Hindi data was selected
automatically. These findings are retrieval-only and do not evaluate STT errors, RAG generation,
frontend behavior, or a production Goa/Konkani document corpus. Re-run the same leakage-safe
protocol when the target corpus and real voice-derived queries become available.
