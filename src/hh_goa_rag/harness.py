"""Structured orchestration for the fully frozen Voice-RAG stack."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from hh_goa_rag.generation import (
    GenerationContext,
    SarvamGeneration,
    SarvamGenerationConfig,
)
from hh_goa_rag.guardrails import (
    GuardrailResponse,
    ReasonCode,
    Route,
    evidence_sufficiency,
    route_input,
    validate_generation,
    validate_transcript,
)
from hh_goa_rag.guardrails.types import STAGE_NAMES
from hh_goa_rag.models import MODEL_SPECS, EmbeddingModel
from hh_goa_rag.retriever import ParentFaissRetriever
from hh_goa_rag.stt.sarvam import SarvamSTT

FROZEN_GENERATION_MODEL = "sarvam-105b"
FROZEN_TOP_K = 10
FROZEN_PROMPT = "strict_context_only"
FROZEN_MAX_OUTPUT_TOKENS = 192
FROZEN_RETRIEVAL = {
    "model": "BAAI/bge-m3",
    "chunking_strategy": "sentence",
    "chunk_size_words": 128,
    "index_engine": "faiss",
    "index_type": "hnsw",
    "m": 32,
    "ef_construction": 200,
    "ef_search": 128,
    "top_k": FROZEN_TOP_K,
}


class Embedder(Protocol):
    def encode_queries(self, texts: list[str]) -> tuple[Any, Any]: ...


class Retriever(Protocol):
    def retrieve(self, query_embedding: Any) -> list[Any]: ...


class Generator(Protocol):
    def generate(
        self, question: str, contexts: list[Any], *, prompt_variant: str
    ) -> Any: ...


class VoiceRAGHarness:
    """Route text/audio through frozen services and deterministic guardrails."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        stt: Any | None = None,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.stt = stt

    @classmethod
    def from_frozen_artifacts(
        cls,
        *,
        retriever_config_path: str | Path = "results/final_retriever_config.json",
        env_path: str | Path = ".env",
        device: str = "auto",
        include_stt: bool = True,
    ) -> VoiceRAGHarness:
        config = json.loads(Path(retriever_config_path).read_text(encoding="utf-8"))
        _assert_frozen_config(config)
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
        embedder = EmbeddingModel(
            MODEL_SPECS[FROZEN_RETRIEVAL["model"]],
            Path(config["model_cache_path"]),
            device=device,
            max_sequence_length=512,
            dtype=dtype,
        )
        retriever = ParentFaissRetriever.load(
            config["index_artifact"],
            config["chunk_artifact"],
            top_k=FROZEN_TOP_K,
            oversample=int(config["search_oversample"]),
        )
        generator = SarvamGeneration.from_env(
            env_path,
            config=SarvamGenerationConfig(
                model=FROZEN_GENERATION_MODEL,
                max_tokens=FROZEN_MAX_OUTPUT_TOKENS,
            ),
        )
        stt = SarvamSTT.from_env(env_path) if include_stt else None
        return cls(embedder=embedder, retriever=retriever, generator=generator, stt=stt)

    def handle_text(self, transcript: object) -> GuardrailResponse:
        return self._handle_transcript(transcript, operation_started=time.perf_counter_ns())

    def handle_audio(self, audio_path: str | Path) -> GuardrailResponse:
        operation_started = time.perf_counter_ns()
        stages = _empty_stages()
        if self.stt is None:
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
            )
        started = time.perf_counter_ns()
        try:
            result = self.stt.transcribe_rest(audio_path)
        except Exception:
            stages["stt"] = _elapsed_ms(started)
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
            )
        stages["stt"] = _elapsed_ms(started)
        stt_metadata = {
            "provider": _field(result, "provider", "sarvam"),
            "model": _field(result, "model", "saaras:v3"),
            "status": _field(result, "status", "error"),
            "error_code": _field(result, "error_code", None),
        }
        if _field(result, "status", "error") != "ok":
            stt_reason = (
                ReasonCode.STT_INVALID_AUDIO
                if _field(result, "error_code", None) == "invalid_audio"
                else ReasonCode.STT_PROVIDER_ERROR
            )
            return _response(
                Route.STT_FAILURE,
                stt_reason,
                stages,
                operation_started,
                metadata={"stt": stt_metadata},
            )
        return self._handle_transcript(
            _field(result, "transcript", ""),
            operation_started=operation_started,
            stages=stages,
            base_metadata={"stt": stt_metadata},
        )

    def close(self) -> None:
        close = getattr(self.embedder, "close", None)
        if callable(close):
            close()

    def _handle_transcript(
        self,
        transcript: object,
        *,
        operation_started: int,
        stages: dict[str, float] | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> GuardrailResponse:
        stages = stages or _empty_stages()
        metadata = dict(base_metadata or {})
        decision_trace: list[dict[str, Any]] = []
        metadata["decision_trace"] = decision_trace
        try:
            started = time.perf_counter_ns()
            validation = validate_transcript(transcript)
            stages["input_validation"] = _elapsed_ms(started)
            decision_trace.append(
                {
                    "stage": "input_validation",
                    "allow": validation.allow,
                    "route": validation.route.value if validation.route else None,
                    "reason_code": (
                        validation.reason_code.value if validation.reason_code else None
                    ),
                }
            )
            if not validation.allow:
                return _response(
                    validation.route or Route.STT_FAILURE,
                    validation.reason_code,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    metadata=metadata,
                )

            started = time.perf_counter_ns()
            route = route_input(validation.normalized_transcript)
            stages["route_check"] = _elapsed_ms(started)
            decision_trace.append(
                {
                    "stage": "route_check",
                    "allow": route.allow,
                    "route": route.route.value if route.route else None,
                    "reason_code": route.reason_code.value if route.reason_code else None,
                }
            )
            if not route.allow:
                return _response(
                    route.route or Route.OFF_TOPIC,
                    route.reason_code,
                    stages,
                    operation_started,
                    transcript=route.normalized_transcript,
                    metadata=metadata,
                )

            started = time.perf_counter_ns()
            vectors, _ = self.embedder.encode_queries([validation.normalized_transcript])
            stages["embedding"] = _elapsed_ms(started)
            started = time.perf_counter_ns()
            contexts = self.retriever.retrieve(vectors[0])
            stages["retrieval"] = _elapsed_ms(started)
            retrieved_ids = tuple(str(_field(item, "parent_id", "")) for item in contexts)
            metadata["retrieved"] = [
                {
                    "rank": index,
                    "parent_id": str(_field(item, "parent_id", "")),
                    "score": float(_field(item, "score", 0.0)),
                }
                for index, item in enumerate(contexts, start=1)
            ]

            started = time.perf_counter_ns()
            sufficiency = evidence_sufficiency(contexts)
            stages["evidence_guardrail"] = _elapsed_ms(started)
            metadata["evidence_decision"] = sufficiency.to_dict()
            decision_trace.append(
                {
                    "stage": "evidence_guardrail",
                    "allow": sufficiency.sufficient,
                    "route": None if sufficiency.sufficient else Route.INSUFFICIENT_CONTEXT.value,
                    "reason_code": (
                        sufficiency.reason_code.value if sufficiency.reason_code else None
                    ),
                    "decision_rule": sufficiency.decision_rule,
                }
            )
            if not sufficiency.sufficient:
                return _response(
                    Route.INSUFFICIENT_CONTEXT,
                    sufficiency.reason_code,
                    stages,
                    operation_started,
                    transcript=validation.normalized_transcript,
                    retrieved_ids=retrieved_ids,
                    metadata=metadata,
                )

            generation_contexts = [
                GenerationContext(
                    parent_id=str(_field(item, "parent_id", "")),
                    chunk_id=str(_field(item, "chunk_id", "")),
                    text=str(_field(item, "text", "")),
                    rank=index,
                    score=float(_field(item, "score", 0.0)),
                )
                for index, item in enumerate(contexts, start=1)
            ]
            started = time.perf_counter_ns()
            generated = self.generator.generate(
                validation.normalized_transcript,
                generation_contexts,
                prompt_variant=FROZEN_PROMPT,
            )
            stages["generation"] = _elapsed_ms(started)
            started = time.perf_counter_ns()
            grounding = validate_generation(generated, generation_contexts)
            stages["grounding_validation"] = _elapsed_ms(started)
            metadata["generation"] = {
                "provider_status": _field(generated, "status", "error"),
                "answer_status": _field(generated, "answer_status", None),
                "error_code": _field(generated, "error_code", None),
            }
            metadata["grounding"] = {
                "valid": grounding.valid,
                "route": grounding.route.value,
                "reason_code": grounding.reason_code.value,
            }
            decision_trace.append(
                {
                    "stage": "grounding_validation",
                    "allow": grounding.valid,
                    "route": grounding.route.value,
                    "reason_code": grounding.reason_code.value,
                }
            )
            return _response(
                grounding.route,
                grounding.reason_code,
                stages,
                operation_started,
                transcript=validation.normalized_transcript,
                answer=grounding.answer,
                retrieved_ids=retrieved_ids,
                citations=grounding.citations,
                metadata=metadata,
            )
        except Exception:
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
                transcript=transcript if isinstance(transcript, str) else "",
                metadata=metadata,
            )


def _assert_frozen_config(config: dict[str, Any]) -> None:
    observed = {
        "model": config.get("model"),
        "chunking_strategy": config.get("chunking", {}).get("strategy"),
        "chunk_size_words": config.get("chunking", {}).get("max_words"),
        "index_engine": config.get("index", {}).get("engine"),
        "index_type": config.get("index", {}).get("index_type"),
        "m": config.get("index", {}).get("m"),
        "ef_construction": config.get("index", {}).get("ef_construction"),
        "ef_search": config.get("index", {}).get("ef_search"),
        "top_k": config.get("top_k"),
    }
    if observed != FROZEN_RETRIEVAL:
        raise RuntimeError(f"Frozen retrieval configuration changed: {observed}")


def _response(
    route: Route,
    reason_code: ReasonCode | None,
    stages: dict[str, float],
    operation_started: int,
    *,
    transcript: str = "",
    answer: str = "",
    retrieved_ids: tuple[str, ...] = (),
    citations: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> GuardrailResponse:
    total_latency = _elapsed_ms(operation_started)
    stage_values = {name: float(stages.get(name, 0.0)) for name in STAGE_NAMES}
    stage_values.update(
        {
            "query_embedding": stage_values["embedding"],
            "vector_search": stage_values["retrieval"],
            "guardrails": sum(
                stage_values[name]
                for name in (
                    "input_validation",
                    "route_check",
                    "evidence_guardrail",
                    "grounding_validation",
                )
            ),
            "total_end_to_end": total_latency,
        }
    )
    return GuardrailResponse(
        route=route,
        answer=answer,
        retrieved_ids=retrieved_ids,
        citations=citations,
        reason_code=reason_code,
        stage_latencies_ms=stage_values,
        total_latency_ms=total_latency,
        transcript=transcript,
        metadata=metadata or {},
    )


def _empty_stages() -> dict[str, float]:
    return {name: 0.0 for name in STAGE_NAMES}


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6
