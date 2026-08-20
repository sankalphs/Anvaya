"""Structured orchestration for the fully frozen Voice-RAG stack."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from hh_goa_rag.generation import (
    GROQ_MODEL,
    GenerationContext,
    GroqGeneration,
    GroqGenerationConfig,
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

FROZEN_GENERATION_MODEL = GROQ_MODEL
FROZEN_TOP_K = 10
FROZEN_PROMPT = "strict_context_only"
FROZEN_MAX_OUTPUT_TOKENS = 128
WARMUP_QUERY = "यूनाइटेड किंगडम में कौन से चार देश शामिल हैं"
WARMUP_PASSAGE = "यूनाइटेड किंगडम में इंग्लैंड, स्कॉटलैंड, वेल्स और उत्तरी आयरलैंड शामिल हैं"
FROZEN_RETRIEVAL = {
    "model": "intfloat/multilingual-e5-small",
    "chunking_strategy": "fixed_words",
    "chunk_size_words": 128,
    "index_engine": "faiss",
    "index_type": "flat_l2",
    "m": None,
    "ef_construction": None,
    "ef_search": None,
    "search_oversample": 20,
    "normalization_method": "float32_l2_v1",
    "top_k": FROZEN_TOP_K,
}


class Embedder(Protocol):
    def encode_queries(self, texts: list[str]) -> tuple[Any, Any]: ...


class Retriever(Protocol):
    def retrieve(self, query_embedding: Any) -> list[Any]: ...


class Generator(Protocol):
    def generate(self, question: str, contexts: list[Any], *, prompt_variant: str) -> Any: ...


class VoiceRAGHarness:
    """Route text/audio through frozen services and deterministic guardrails."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        stt: Any | None = None,
        orchestrator: str = "native",
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.stt = stt
        if orchestrator not in {"native", "langgraph"}:
            raise ValueError("orchestrator must be 'native' or 'langgraph'")
        self.orchestrator = orchestrator
        self._graph = None
        if orchestrator == "langgraph":
            from hh_goa_rag.orchestration import build_langgraph

            self._graph = build_langgraph(self._handle_transcript)

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
            _portable_path(config["model_cache_path"]),
            device=device,
            max_sequence_length=512,
            dtype=dtype,
        )
        retriever = ParentFaissRetriever.load(
            _portable_path(config["index_artifact"]),
            _portable_path(config["chunk_artifact"]),
            top_k=FROZEN_TOP_K,
            oversample=int(config["search_oversample"]),
        )
        generator = GroqGeneration.from_env(
            env_path,
            config=GroqGenerationConfig(
                model=FROZEN_GENERATION_MODEL,
                max_tokens=FROZEN_MAX_OUTPUT_TOKENS,
            ),
        )
        stt = SarvamSTT.from_env(env_path) if include_stt else None
        orchestrator = os.getenv("HH_RAG_ORCHESTRATOR", "native").strip().lower()
        generator.warm_up()
        embedder.warm_up(WARMUP_QUERY, WARMUP_PASSAGE, rounds=1)
        return cls(
            embedder=embedder,
            retriever=retriever,
            generator=generator,
            stt=stt,
            orchestrator=orchestrator,
        )

    def handle_text(
        self,
        transcript: object,
        *,
        language_code: str | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        return self._dispatch_transcript(
            transcript,
            operation_started=time.perf_counter_ns(),
            base_metadata=(
                {"input_language": language_code} if language_code is not None else None
            ),
            on_stage=on_stage,
        )

    def handle_audio(
        self,
        audio_path: str | Path,
        *,
        language_code: str = "hi-IN",
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        operation_started = time.perf_counter_ns()
        stages = _empty_stages()
        if self.stt is None:
            return _response(
                Route.SYSTEM_ERROR,
                ReasonCode.SYSTEM_COMPONENT_ERROR,
                stages,
                operation_started,
            )
        _notify_stage(on_stage, "Transcribing")
        started = time.perf_counter_ns()
        try:
            result = self.stt.transcribe_rest(audio_path, language_code=language_code)
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
            "requested_language": language_code,
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
        return self._dispatch_transcript(
            _field(result, "transcript", ""),
            operation_started=operation_started,
            stages=stages,
            base_metadata={"stt": stt_metadata, "input_language": language_code},
            on_stage=on_stage,
        )

    def _dispatch_transcript(
        self,
        transcript: object,
        *,
        operation_started: int,
        stages: dict[str, float] | None = None,
        base_metadata: dict[str, Any] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        if self._graph is None:
            response = self._handle_transcript(
                transcript,
                operation_started=operation_started,
                stages=stages,
                base_metadata=base_metadata,
                on_stage=on_stage,
            )
        else:
            result = self._graph.invoke(
                {
                    "transcript": transcript,
                    "operation_started": operation_started,
                    "stages": stages,
                    "base_metadata": base_metadata,
                    "on_stage": on_stage,
                }
            )
            response = result["response"]
        response.metadata.setdefault("orchestrator", self.orchestrator)
        return response

    def close(self) -> None:
        close = getattr(self.embedder, "close", None)
        if callable(close):
            close()
        close = getattr(self.generator, "close", None)
        if callable(close):
            close()

    def _handle_transcript(
        self,
        transcript: object,
        *,
        operation_started: int,
        stages: dict[str, float] | None = None,
        base_metadata: dict[str, Any] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> GuardrailResponse:
        stages = stages or _empty_stages()
        metadata = dict(base_metadata or {})
        decision_trace: list[dict[str, Any]] = []
        metadata["decision_trace"] = decision_trace
        try:
            _notify_stage(on_stage, "Checking query")
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

            _notify_stage(on_stage, "Retrieving evidence")
            started = time.perf_counter_ns()
            vectors, _ = self.embedder.encode_queries([validation.normalized_transcript])
            stages["embedding"] = _elapsed_ms(started)
            started = time.perf_counter_ns()
            contexts = list(self.retriever.retrieve(vectors[0]))[:FROZEN_TOP_K]
            stages["retrieval"] = _elapsed_ms(started)
            retrieved_ids = tuple(str(_field(item, "parent_id", "")) for item in contexts)
            metadata["retrieved"] = [
                {
                    "rank": index,
                    "parent_id": str(_field(item, "parent_id", "")),
                    "chunk_id": str(_field(item, "chunk_id", "")),
                    "score": float(_field(item, "score", 0.0)),
                    "text": str(_field(item, "text", "")),
                }
                for index, item in enumerate(contexts, start=1)
            ]

            started = time.perf_counter_ns()
            sufficiency = evidence_sufficiency(contexts, query=validation.normalized_transcript)
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
            _notify_stage(on_stage, "Generating answer")
            started = time.perf_counter_ns()
            generated = self.generator.generate(
                validation.normalized_transcript,
                generation_contexts,
                prompt_variant=FROZEN_PROMPT,
            )
            stages["generation"] = _elapsed_ms(started)
            _notify_stage(on_stage, "Validating grounding")
            started = time.perf_counter_ns()
            grounding = validate_generation(generated, generation_contexts)
            stages["grounding_validation"] = _elapsed_ms(started)
            metadata["generation"] = {
                "provider": _field(generated, "provider", "groq"),
                "model": _field(generated, "model", FROZEN_GENERATION_MODEL),
                "provider_status": _field(generated, "status", "error"),
                "answer_status": _field(generated, "answer_status", None),
                "error_code": _field(generated, "error_code", None),
                "error_message": _field(generated, "error_message", None),
                "http_status": _field(generated, "http_status", None),
                "latency_ms": _field(generated, "latency_ms", None),
                "time_to_first_token_ms": _field(generated, "time_to_first_token_ms", None),
                "prompt_tokens": _field(generated, "prompt_tokens", None),
                "output_tokens": _field(generated, "output_tokens", None),
                "tokens_per_second": _field(generated, "tokens_per_second", None),
                "provider_latency_ms": _field(generated, "provider_latency_ms", None),
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
            if (
                not grounding.valid
                and grounding.reason_code != ReasonCode.MODEL_INSUFFICIENT_CONTEXT
            ):
                fallback = _extractive_kb_fallback(generation_contexts)
                if fallback is not None:
                    fallback_answer, fallback_parent_id = fallback
                    metadata["generation"]["fallback"] = "extractive_kb"
                    metadata["generation"]["fallback_reason"] = grounding.reason_code.value
                    metadata["grounding"] = {
                        "valid": True,
                        "route": Route.ANSWER.value,
                        "reason_code": ReasonCode.ANSWER_GROUNDED.value,
                    }
                    decision_trace.append(
                        {
                            "stage": "kb_fallback",
                            "allow": True,
                            "route": Route.ANSWER.value,
                            "reason_code": ReasonCode.ANSWER_GROUNDED.value,
                        }
                    )
                    return _response(
                        Route.ANSWER,
                        ReasonCode.ANSWER_GROUNDED,
                        stages,
                        operation_started,
                        transcript=validation.normalized_transcript,
                        answer=fallback_answer,
                        retrieved_ids=retrieved_ids,
                        citations=(fallback_parent_id,),
                        metadata=metadata,
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
        "search_oversample": config.get("search_oversample"),
        "normalization_method": config.get("normalization_method"),
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


def _extractive_kb_fallback(
    contexts: list[GenerationContext],
) -> tuple[str, str] | None:
    """Return a bounded verbatim KB sentence when the model output is unusable.

    This is deliberately extractive: it cannot introduce model knowledge, and it
    is only reachable after retrieval sufficiency has already passed.
    """

    for context in contexts[:3]:
        text = str(context.text).strip()
        if not text:
            continue
        sentence = re.split(r"(?<=[.!?।])\s+", text, maxsplit=1)[0].strip()
        answer = sentence or text
        if answer:
            return answer[:500], context.parent_id
    return None


def _elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1e6


def _notify_stage(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _portable_path(value: str | Path) -> Path:
    """Interpret persisted repository-relative paths on Windows and Linux."""
    return Path(str(value).replace("\\", "/"))
