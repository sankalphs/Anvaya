"""Deterministic validation of schema-constrained generation output."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import ReasonCode, Route

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_GROUNDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    "और", "का", "की", "के", "को", "में", "से", "है", "हैं", "यह", "एक",
}


@dataclass(frozen=True)
class GroundingDecision:
    valid: bool
    route: Route
    reason_code: ReasonCode
    answer: str
    citations: tuple[str, ...]


def validate_generation(result: Any, contexts: Sequence[Any]) -> GroundingDecision:
    allowed_ids = {str(_field(context, "parent_id", "")) for context in contexts}
    transport_status = str(_field(result, "status", "error"))
    if transport_status != "ok":
        error_code = str(_field(result, "error_code", ""))
        reason = (
            ReasonCode.GENERATION_SCHEMA_INVALID
            if error_code == "invalid_structured_output"
            else ReasonCode.GENERATION_PROVIDER_ERROR
        )
        return GroundingDecision(False, Route.SYSTEM_ERROR, reason, "", ())

    answer_status = str(_field(result, "answer_status", ""))
    answer = str(_field(result, "answer", "") or "").strip()
    evidence_ids = tuple(dict.fromkeys(str(item) for item in _field(result, "evidence_ids", ())))
    raw_output = str(_field(result, "raw_output", "") or "")
    if raw_output:
        raw = _parse_raw(raw_output)
        if raw is None or (
            raw["status"] != answer_status
            or raw["answer"].strip() != answer
            or tuple(dict.fromkeys(raw["evidence_ids"])) != evidence_ids
        ):
            return GroundingDecision(
                False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_SCHEMA_INVALID, "", ()
            )
    diagnostics = _field(result, "diagnostics", {}) or {}
    if diagnostics.get("schema_valid") is False:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_SCHEMA_INVALID, "", ()
        )

    unknown = set(evidence_ids) - allowed_ids
    if unknown:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_UNKNOWN_CITATION, "", ()
        )
    if answer_status == "INSUFFICIENT_CONTEXT":
        if answer or evidence_ids:
            return GroundingDecision(
                False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_INVALID_REFUSAL, "", ()
            )
        return GroundingDecision(
            True,
            Route.INSUFFICIENT_CONTEXT,
            ReasonCode.MODEL_INSUFFICIENT_CONTEXT,
            "",
            (),
        )
    if answer_status != "ANSWER":
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_SCHEMA_INVALID, "", ()
        )
    if not answer:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_EMPTY_ANSWER, "", ()
        )
    if not evidence_ids:
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_MISSING_CITATION, "", ()
        )
    cited_text = " ".join(
        str(_field(context, "text", ""))
        for context in contexts
        if str(_field(context, "parent_id", "")) in set(evidence_ids)
    )
    if not _answer_overlaps_evidence(answer, cited_text):
        return GroundingDecision(
            False, Route.SYSTEM_ERROR, ReasonCode.GENERATION_UNGROUNDED_ANSWER, "", ()
        )
    return GroundingDecision(
        True, Route.ANSWER, ReasonCode.ANSWER_GROUNDED, answer, evidence_ids
    )


def _answer_overlaps_evidence(answer: str, evidence: str, *, minimum: float = 0.2) -> bool:
    """Require informative answer tokens to occur in the cited evidence.

    This is intentionally a conservative lexical gate. It is not a substitute for
    human/entailment evaluation, but it prevents a valid citation ID from acting as
    a blanket approval for an unrelated answer.
    """
    def tokens(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return {
            token
            for token in _TOKEN_RE.findall(normalized)
            if token not in _GROUNDING_STOPWORDS
        }

    answer_tokens = tokens(answer)
    evidence_tokens = tokens(evidence)
    if not answer_tokens or not evidence_tokens:
        return False
    overlap = len(answer_tokens & evidence_tokens) / len(answer_tokens)
    return overlap >= minimum


def _parse_raw(raw_output: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    valid = (
        isinstance(value, dict)
        and set(value) == {"status", "answer", "evidence_ids"}
        and value["status"] in {"ANSWER", "INSUFFICIENT_CONTEXT"}
        and isinstance(value["answer"], str)
        and isinstance(value["evidence_ids"], list)
        and all(isinstance(item, str) for item in value["evidence_ids"])
    )
    return value if valid else None


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
