"""Development-selected retrieval sufficiency signals for the frozen Top-10 retriever."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import ReasonCode

TOP_SCORE_THRESHOLD = 0.67
CONSISTENCY_RESCUE_FLOOR = 0.64
TOP_TWO_MAX_GAP = 0.005
TOP_TO_FIFTH_MIN_SPREAD = 0.12


@dataclass(frozen=True)
class RetrievalSignals:
    sufficient: bool
    reason_code: ReasonCode | None
    top_score: float | None
    top_two_gap: float | None
    top_to_fifth_spread: float | None
    top_three_mean: float | None
    decision_rule: str

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "sufficient": self.sufficient,
            "reason_code": self.reason_code.value if self.reason_code else None,
            "top_score": self.top_score,
            "top_two_gap": self.top_two_gap,
            "top_to_fifth_spread": self.top_to_fifth_spread,
            "top_three_mean": self.top_three_mean,
            "decision_rule": self.decision_rule,
        }


def evidence_sufficiency(
    contexts: Sequence[Any],
    *,
    top_score_threshold: float = TOP_SCORE_THRESHOLD,
) -> RetrievalSignals:
    scores = [float(_field(context, "score")) for context in contexts]
    if not scores:
        return RetrievalSignals(
            False, ReasonCode.RETRIEVAL_EMPTY, None, None, None, None, "empty"
        )
    top = scores[0]
    gap = top - scores[1] if len(scores) >= 2 else None
    spread = top - scores[4] if len(scores) >= 5 else None
    mean3 = sum(scores[:3]) / min(3, len(scores))
    if top >= top_score_threshold:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "top_score")
    corroborated = (
        top >= CONSISTENCY_RESCUE_FLOOR and gap is not None and gap <= TOP_TWO_MAX_GAP
    )
    if corroborated:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "top_two_corroboration")
    dominant = (
        top >= CONSISTENCY_RESCUE_FLOOR
        and spread is not None
        and spread >= TOP_TO_FIFTH_MIN_SPREAD
    )
    if dominant:
        return RetrievalSignals(True, None, top, gap, spread, mean3, "top_to_fifth_spread")
    return RetrievalSignals(
        False,
        ReasonCode.RETRIEVAL_LOW_CONFIDENCE,
        top,
        gap,
        spread,
        mean3,
        "below_threshold_without_consistency_rescue",
    )


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name)
