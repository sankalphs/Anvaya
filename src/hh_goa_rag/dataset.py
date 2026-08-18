"""MSMARCO-XI discovery and leakage-safe evaluation artifact creation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.io import write_json, write_jsonl

LANGUAGE_PREFIXES: dict[str, str] = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}
DEFAULT_LANGUAGE = "hi"


@dataclass(frozen=True)
class DatasetResolution:
    repository: str
    revision: str
    language: str
    train_file: str | None
    validation_file: str | None
    available_languages: tuple[str, ...]


def discover_resolution(
    repository: str,
    requested_language: str = "auto",
    revision: str | None = None,
    *,
    api: HfApi | None = None,
) -> DatasetResolution:
    """Resolve current Hub files to an immutable revision and language.

    The dataset's legacy loading script is no longer honored by recent versions of
    ``datasets``. File discovery is therefore based on the actual parquet inventory.
    """
    info = (api or HfApi()).dataset_info(repository, revision=revision, files_metadata=True)
    files = {sibling.rfilename for sibling in info.siblings}
    by_language: dict[str, dict[str, str]] = {}
    for language, prefix in LANGUAGE_PREFIXES.items():
        for split, folder, suffix in (
            ("train", "train", "train"),
            ("validation", "validation", "val"),
        ):
            expected = f"{folder}/{prefix}{suffix}.parquet"
            if expected in files:
                by_language.setdefault(language, {})[split] = expected

    available = tuple(sorted(by_language))
    if not available:
        raise RuntimeError(f"No recognized MSMARCO-XI parquet files found in {repository}")
    language = DEFAULT_LANGUAGE if requested_language == "auto" else requested_language
    if language not in by_language:
        raise ValueError(f"Language {language!r} unavailable; discovered {list(available)}")
    selected = by_language[language]
    return DatasetResolution(
        repository=repository,
        revision=info.sha,
        language=language,
        train_file=selected.get("train"),
        validation_file=selected.get("validation"),
        available_languages=available,
    )


def hf_parquet_uri(resolution: DatasetResolution, split: str) -> str:
    filename = resolution.train_file if split == "train" else resolution.validation_file
    if filename is None:
        raise ValueError(f"{resolution.language!r} has no {split!r} parquet file")
    return f"hf://datasets/{resolution.repository}@{resolution.revision}/{filename}"


def download_split(
    resolution: DatasetResolution, split: str, cache_dir: str | Path
) -> Path:
    """Download one pinned parquet to a caller-managed local directory."""
    filename = resolution.train_file if split == "train" else resolution.validation_file
    if filename is None:
        raise ValueError(f"{resolution.language!r} has no {split!r} parquet file")
    local_dir = Path(cache_dir) / "raw" / resolution.revision
    return Path(
        hf_hub_download(
            repo_id=resolution.repository,
            filename=filename,
            repo_type="dataset",
            revision=resolution.revision,
            local_dir=local_dir,
        )
    )


def iter_parquet_rows(path: str | Path, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    """Read only retrieval-relevant columns, bounded by an Arrow batch."""
    columns = ["source_lang", "target_lang", "query_id", "query", "passages"]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def _stable_unit_interval(seed: int, split: str, query_id: Any) -> float:
    digest = hashlib.sha256(f"{seed}:{split}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    split: str,
    acceptance_rate: float,
    max_rows_scanned: int,
) -> tuple[list[dict[str, Any]], int]:
    """Choose a deterministic hash sample without relying on stream shuffle buffers."""
    if not 0 < acceptance_rate <= 1:
        raise ValueError("acceptance_rate must be in (0, 1]")
    selected: list[dict[str, Any]] = []
    scanned = 0
    for row in rows:
        scanned += 1
        labels = row.get("passages", {}).get("is_selected") or []
        usable = bool(row.get("query")) and any(int(label) == 1 for label in labels)
        score = _stable_unit_interval(seed, split, row.get("query_id"))
        if usable and score < acceptance_rate:
            selected.append(row)
            if len(selected) == count:
                break
        if scanned >= max_rows_scanned:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Only selected {len(selected)}/{count} usable {split} rows after scanning "
            f"{scanned}; raise acceptance_rate or max_rows_scanned_multiplier"
        )
    return selected, scanned


def _passage_id(language: str, text: str) -> str:
    digest = hashlib.sha256(f"{language}\0{text}".encode()).hexdigest()[:24]
    return f"p-{digest}"


def materialize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    language: str,
    passage_field: str,
    query_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool passages and retain query-to-parent-passage relevance labels."""
    corpus_by_id: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen_queries:
            continue
        passages = row["passages"]
        texts = passages[passage_field]
        labels = passages["is_selected"]
        if len(texts) != len(labels):
            raise ValueError(f"Passage/label length mismatch for query {query_id}")
        relevant: set[str] = set()
        candidate_ids: list[str] = []
        for text, label in zip(texts, labels, strict=True):
            normalized_text = " ".join(str(text).split())
            if not normalized_text:
                continue
            passage_id = _passage_id(language, normalized_text)
            candidate_ids.append(passage_id)
            corpus_by_id.setdefault(
                passage_id,
                {"passage_id": passage_id, "language": language, "text": normalized_text},
            )
            if int(label) == 1:
                relevant.add(passage_id)
        if not relevant:
            continue
        seen_queries.add(query_id)
        queries.append(
            {
                "query_id": query_id,
                "language": language,
                "text": " ".join(str(row[query_field]).split()),
                "candidate_parent_ids": sorted(set(candidate_ids)),
            }
        )
        for passage_id in sorted(relevant):
            qrels.append({"query_id": query_id, "passage_id": passage_id, "relevance": 1})

    corpus = sorted(corpus_by_id.values(), key=lambda item: item["passage_id"])
    queries.sort(key=lambda item: item["query_id"])
    qrels.sort(key=lambda item: (item["query_id"], item["passage_id"]))
    return corpus, queries, qrels


def inspect_schema(row: dict[str, Any]) -> dict[str, Any]:
    passages = row.get("passages") or {}
    return {
        "top_level_fields": sorted(row),
        "passage_fields": sorted(passages),
        "passage_count": len(passages.get("is_selected") or []),
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
    }


def prepare_dataset(config: dict[str, Any], *, force: bool = False) -> Path:
    """Download/stream, inspect, sample, and cache development and sealed test data."""
    dataset_config = config["dataset"]
    seed = int(config["experiment"]["seed"])
    resolution = discover_resolution(
        dataset_config["repository"],
        dataset_config.get("language", "auto"),
        dataset_config.get("revision"),
    )
    artifact_identity = {
        "resolution": asdict(resolution),
        "seed": seed,
        "dev_queries": int(dataset_config["dev_queries"]),
        "test_queries": int(dataset_config["test_queries"]),
        "sampling_acceptance_rate": float(dataset_config["sampling_acceptance_rate"]),
        "max_rows_scanned_multiplier": int(dataset_config["max_rows_scanned_multiplier"]),
        "passage_field": dataset_config["passage_field"],
        "query_field": dataset_config["query_field"],
    }
    output_dir = Path(dataset_config["output_dir"]) / stable_fingerprint(artifact_identity)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return output_dir

    cache_dir = Path(config["cache"]["huggingface"])
    split_specs = {
        "dev": (dataset_config["dev_source_split"], int(dataset_config["dev_queries"])),
        "test": (dataset_config["test_source_split"], int(dataset_config["test_queries"])),
    }
    split_stats: dict[str, Any] = {}
    schema: dict[str, Any] | None = None
    for output_split, (source_split, count) in split_specs.items():
        parquet_path = download_split(resolution, source_split, cache_dir)
        try:
            max_scanned = count * int(dataset_config["max_rows_scanned_multiplier"])
            selected, scanned = select_rows(
                iter_parquet_rows(parquet_path),
                count=count,
                seed=seed,
                split=source_split,
                acceptance_rate=float(dataset_config["sampling_acceptance_rate"]),
                max_rows_scanned=max_scanned,
            )
        finally:
            if not bool(dataset_config.get("retain_raw_parquet", False)):
                parquet_path.unlink(missing_ok=True)
        if schema is None:
            schema = inspect_schema(selected[0])
        corpus, queries, qrels = materialize_rows(
            selected,
            language=resolution.language,
            passage_field=dataset_config["passage_field"],
            query_field=dataset_config["query_field"],
        )
        write_jsonl(output_dir / f"{output_split}_corpus.jsonl", corpus)
        write_jsonl(output_dir / f"{output_split}_queries.jsonl", queries)
        write_jsonl(output_dir / f"{output_split}_qrels.jsonl", qrels)
        split_stats[output_split] = {
            "source_split": source_split,
            "rows_scanned": scanned,
            "queries": len(queries),
            "unique_parent_passages": len(corpus),
            "qrels": len(qrels),
        }

    manifest = {
        "format_version": 1,
        "artifact_fingerprint": output_dir.name,
        "artifact_identity": artifact_identity,
        "dataset_schema": schema,
        "splits": split_stats,
        "leakage_policy": {
            "development": "A deterministic sample of the upstream train parquet.",
            "test": "A deterministic sample of the upstream validation parquet; final winner only.",
            "gold_unit": (
                "Parent passage ID; retrieved chunks are mapped to their parent before scoring."
            ),
        },
    }
    write_json(manifest_path, manifest)
    return output_dir


def read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
