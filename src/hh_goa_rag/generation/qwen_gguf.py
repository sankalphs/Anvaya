"""Local Qwen3.5 Q4 GGUF answer generation for the ZeroGPU Space."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .prompts import PromptVariant, build_messages
from .sarvam import GenerationContext, GenerationResult, _parse_answer

QWEN_GGUF_REPOSITORY = "ggml-org/Qwen3.5-0.8B-GGUF"
QWEN_GGUF_FILENAME = "Qwen3.5-0.8B-Q4_0.gguf"
QWEN_GGUF_MODEL = f"{QWEN_GGUF_REPOSITORY}/{QWEN_GGUF_FILENAME}"

_MODEL: Any | None = None
_MODEL_PATH = ""
_MODEL_RUNTIME = "uninitialized"
_MODEL_LOCK = threading.Lock()
_GPU_ENTRY: Any | None = None


@dataclass(frozen=True)
class QwenGGUFGenerationConfig:
    model: str = QWEN_GGUF_MODEL
    max_tokens: int = 48
    context_size: int = 2048
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.context_size <= 0 or self.timeout_s <= 0:
            raise ValueError("max_tokens, context_size, and timeout_s must be positive")


def set_model_path(path: str) -> None:
    """Set the Space-local GGUF path before the first GPU request."""
    global _MODEL_PATH
    _MODEL_PATH = path


def _generate_on_gpu(
    messages: list[dict[str, str]], config: QwenGGUFGenerationConfig
) -> dict[str, Any]:
    """Run one deterministic Qwen Q4 generation inside a ZeroGPU allocation."""
    global _MODEL, _MODEL_RUNTIME

    if not _MODEL_PATH:
        raise RuntimeError("QWEN_GGUF_PATH is not configured")

    from llama_cpp import Llama

    with _MODEL_LOCK:
        load_started = time.perf_counter_ns()
        model_load_ms = 0.0
        if _MODEL is None:
            try:
                _MODEL = Llama(
                    model_path=_MODEL_PATH,
                    n_ctx=config.context_size,
                    n_batch=1024,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                _MODEL_RUNTIME = "llama.cpp/cuda"
            except Exception as error:
                print(f"Qwen CUDA load failed; using CPU fallback: {error!r}", flush=True)
                _MODEL = Llama(
                    model_path=_MODEL_PATH,
                    n_ctx=config.context_size,
                    n_batch=1024,
                    n_gpu_layers=0,
                    verbose=False,
                )
                _MODEL_RUNTIME = "llama.cpp/cpu-fallback"
            model_load_ms = _elapsed_ms(load_started)

        prompt_started = time.perf_counter_ns()
        generation_started = time.perf_counter_ns()
        pieces: list[str] = []
        first_token_ms: float | None = None
        request = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            pieces, first_token_ms = _collect_stream(
                _MODEL, request, generation_started, disable_thinking=True
            )
        except Exception as error:
            if _MODEL_RUNTIME != "llama.cpp/cuda":
                raise
            print(f"Qwen CUDA generation failed; using CPU fallback: {error!r}", flush=True)
            _MODEL = Llama(
                model_path=_MODEL_PATH,
                n_ctx=config.context_size,
                n_batch=1024,
                n_gpu_layers=0,
                verbose=False,
            )
            _MODEL_RUNTIME = "llama.cpp/cpu-fallback"
            pieces, first_token_ms = _collect_stream(
                _MODEL, request, generation_started, disable_thinking=False
            )

        raw_output = "".join(pieces).strip()
        return {
            "raw_output": raw_output,
            "model_load_ms": model_load_ms,
            "prompt_ms": _elapsed_ms(prompt_started),
            "generation_ms": _elapsed_ms(generation_started),
            "total_ms": _elapsed_ms(load_started),
            "time_to_first_token_ms": first_token_ms,
            "output_tokens": len(_MODEL.tokenize(raw_output.encode("utf-8"), add_bos=False)),
        }


def _generate_on_cpu(
    messages: list[dict[str, str]], config: QwenGGUFGenerationConfig
) -> dict[str, Any]:
    """Run the same GGUF request without relying on a ZeroGPU allocation."""
    global _MODEL, _MODEL_RUNTIME

    if not _MODEL_PATH:
        raise RuntimeError("QWEN_GGUF_PATH is not configured")

    from llama_cpp import Llama

    with _MODEL_LOCK:
        load_started = time.perf_counter_ns()
        _MODEL = Llama(
            model_path=_MODEL_PATH,
            n_ctx=config.context_size,
            n_batch=1024,
            n_gpu_layers=0,
            verbose=False,
        )
        _MODEL_RUNTIME = "llama.cpp/cpu-fallback"
        model_load_ms = _elapsed_ms(load_started)
        prompt_started = time.perf_counter_ns()
        generation_started = time.perf_counter_ns()
        request = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        pieces, first_token_ms = _collect_stream(
            _MODEL, request, generation_started, disable_thinking=False
        )
        raw_output = "".join(pieces).strip()
        return {
            "raw_output": raw_output,
            "model_load_ms": model_load_ms,
            "prompt_ms": _elapsed_ms(prompt_started),
            "generation_ms": _elapsed_ms(generation_started),
            "total_ms": _elapsed_ms(load_started),
            "time_to_first_token_ms": first_token_ms,
            "output_tokens": len(_MODEL.tokenize(raw_output.encode("utf-8"), add_bos=False)),
        }


def _collect_stream(
    model: Any,
    request: dict[str, Any],
    generation_started: int,
    *,
    disable_thinking: bool,
) -> tuple[list[str], float | None]:
    try:
        stream = model.create_chat_completion(
            **request,
            chat_template_kwargs={"enable_thinking": False},
        ) if disable_thinking else model.create_chat_completion(**request)
    except TypeError:
        stream = model.create_chat_completion(**request)
    if not request.get("stream", True):
        choice = (stream.get("choices") or [{}])[0]
        content = str((choice.get("message") or {}).get("content") or "")
        return ([content] if content else []), None
    pieces: list[str] = []
    first_token_ms: float | None = None
    for event in stream:
        choice = (event.get("choices") or [{}])[0]
        content = str((choice.get("delta") or {}).get("content") or "")
        if content:
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(generation_started)
            pieces.append(content)
    return pieces, first_token_ms


class QwenGGUFGeneration:
    """Generate grounded answers with the exact Qwen3.5 0.8B Q4_0 GGUF."""

    def __init__(self, *, config: QwenGGUFGenerationConfig | None = None) -> None:
        self.config = config or QwenGGUFGenerationConfig()
        set_model_path(os.environ.get("QWEN_GGUF_PATH", ""))
        _configure_gpu_entry()

    @classmethod
    def from_env(
        cls,
        *,
        config: QwenGGUFGenerationConfig | None = None,
    ) -> QwenGGUFGeneration:
        return cls(config=config)

    def warm_up(self) -> None:
        """Keep startup CPU-only; ZeroGPU is acquired only for a real answer."""

    def close(self) -> None:
        """Release the worker-local model if the host asks the harness to close."""
        global _MODEL
        _MODEL = None

    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext | dict[str, Any]],
        *,
        prompt_variant: PromptVariant = "structured_evidence_ids",
        language_code: str | None = None,
    ) -> GenerationResult:
        prompt_contexts = list(contexts)[:3]
        messages = build_messages(
            question,
            prompt_contexts,
            variant=prompt_variant,
            language_code=language_code,
        )
        allowed_ids = {str(_field(context, "parent_id")) for context in prompt_contexts}
        started = time.perf_counter_ns()
        try:
            observation = _gpu_entry()(messages, self.config)
        except Exception as error:
            # A ZeroGPU decorator failure can happen before _generate_on_gpu
            # runs, so the in-function CUDA→CPU fallback is not sufficient.
            # Retry the request directly on CPU before returning a provider
            # error that would trigger a wrong-language evidence fallback.
            print(f"Qwen ZeroGPU request failed; retrying on CPU: {error!r}", flush=True)
            try:
                observation = _generate_on_cpu(messages, self.config)
            except Exception:
                raise error from None
        try:
            raw_output = str(observation.get("raw_output") or "")
            cleaned_output = _clean_model_output(raw_output)
            parsed, diagnostics = _parse_answer(cleaned_output, allowed_ids)
            checkpoints = {
                "qwen_model_load_ms": observation.get("model_load_ms"),
                "qwen_prompt_ms": observation.get("prompt_ms"),
                "qwen_generation_ms": observation.get("generation_ms"),
                "qwen_total_ms": observation.get("total_ms"),
                "qwen_time_to_first_token_ms": observation.get("time_to_first_token_ms"),
                "qwen_output_tokens": observation.get("output_tokens"),
            "qwen_runtime": _MODEL_RUNTIME,
                "qwen_model_file": QWEN_GGUF_FILENAME,
            }
            if parsed is None:
                return self._result(
                    started,
                    status="error",
                    code="invalid_structured_output",
                    message=str(diagnostics.get("parse_error", "invalid JSON")),
                    raw_output=raw_output,
                    diagnostics={**diagnostics, "checkpoints": checkpoints},
                    ttft_ms=observation.get("time_to_first_token_ms"),
                    output_tokens=observation.get("output_tokens"),
                )
            return self._result(
                started,
                status="ok",
                answer_status=parsed["status"],
                answer=parsed["answer"],
                evidence_ids=tuple(parsed["evidence_ids"]),
                raw_output=raw_output,
                diagnostics={**diagnostics, "checkpoints": checkpoints},
                ttft_ms=observation.get("time_to_first_token_ms"),
                output_tokens=observation.get("output_tokens"),
            )
        except Exception as error:
            return self._result(
                started,
                status="error",
                code=type(error).__name__,
                message=str(error)[:500],
            )

    def _result(
        self,
        started: int,
        *,
        status: str,
        code: str | None = None,
        message: str | None = None,
        raw_output: str = "",
        answer_status: str | None = None,
        answer: str = "",
        evidence_ids: tuple[str, ...] = (),
        diagnostics: dict[str, Any] | None = None,
        ttft_ms: float | None = None,
        output_tokens: int | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            provider="qwen-gguf",
            model=self.config.model,
            status=status,  # type: ignore[arg-type]
            answer_status=answer_status,  # type: ignore[arg-type]
            answer=answer,
            evidence_ids=evidence_ids,
            raw_output=raw_output,
            latency_ms=_elapsed_ms(started),
            time_to_first_token_ms=ttft_ms,
            prompt_tokens=None,
            output_tokens=output_tokens,
            total_tokens=None,
            finish_reason=None,
            attempts=1,
            error_code=code,
            error_message=message,
            diagnostics=diagnostics or {},
        )


def _clean_model_output(raw_output: str) -> str:
    cleaned = re.sub(
        r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name, "")
    return getattr(value, name, "")


def _gpu_entry() -> Any:
    if _GPU_ENTRY is not None:
        return _GPU_ENTRY
    return _generate_on_gpu


def _configure_gpu_entry() -> None:
    """Use ZeroGPU inside Spaces without requiring a non-standard env var."""

    global _GPU_ENTRY
    if _GPU_ENTRY is not None:
        return
    if not (os.environ.get("SPACE_ID") or os.environ.get("SPACES_ZERO_GPU") == "1"):
        return
    import spaces

    _GPU_ENTRY = spaces.GPU(duration=45)(_generate_on_gpu)


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000
