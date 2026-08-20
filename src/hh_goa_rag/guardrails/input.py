"""Cheap transcript validation and policy routing without model calls."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .types import ReasonCode, Route

MAX_TRANSCRIPT_CHARS = 2_000
_FILLERS = {
    "ah",
    "hmm",
    "hm",
    "uh",
    "um",
    "अं",
    "अह",
    "हम्म",
    "हां",
    "हाँ",
}

_UNSAFE_RULES: tuple[tuple[ReasonCode, tuple[str, ...]], ...] = (
    (
        ReasonCode.UNSAFE_CREDENTIAL_THEFT,
        (
            "password चुरा",
            "पासवर्ड चुरा",
            "steal password",
            "credential theft",
            "steal the password",
            "how do i steal the password",
            "पासवर्ड चोरी",
            "पासवर्ड कैसे चुर",
        ),
    ),
    (
        ReasonCode.UNSAFE_WEAPONS,
        (
            "untraceable weapon",
            "build a weapon",
            "make a bomb",
            "how to make a bomb",
            "हथियार बना",
            "बम बना",
            "बम कैसे बना",
        ),
    ),
    (
        ReasonCode.UNSAFE_PHYSICAL_HARM,
        (
            "नुकसान पहुँचाने",
            "जहर कैसे",
            "ज़हर कैसे",
            "poison someone",
            "how to poison someone",
            "kill someone",
        ),
    ),
    (
        ReasonCode.UNSAFE_HATE,
        ("नफरत भरा", "hate message", "अपमानजनक नफरत", "ethnic slur"),
    ),
)

_OFF_TOPIC_RULES: tuple[tuple[ReasonCode, tuple[str, ...]], ...] = (
    (
        ReasonCode.OFF_TOPIC_CREATIVE_WRITING,
        (
            "कविता लिख",
            "कहानी लिख",
            "write a poem",
            "write a story",
            "tell me a joke",
            "make me laugh",
            "चुटकुला सुनाओ",
        ),
    ),
    (
        ReasonCode.OFF_TOPIC_LIVE_INFORMATION,
        (
            "लाइव क्रिकेट",
            "live cricket",
            "live score",
            "आज का स्कोर",
            "what is the weather",
            "weather today",
            "current weather",
            "who won the 2026 world cup",
            "आज का मौसम",
        ),
    ),
    (
        ReasonCode.OFF_TOPIC_TRANSACTION,
        ("book me a taxi", "टैक्सी बुक", "order me", "मेरे लिए बुक"),
    ),
    (
        ReasonCode.OFF_TOPIC_RECIPE,
        ("रेसिपी", "recipe", "केक बनाने", "cake बन"),
    ),
)


@dataclass(frozen=True)
class InputDecision:
    allow: bool
    normalized_transcript: str
    route: Route | None = None
    reason_code: ReasonCode | None = None


def validate_transcript(transcript: object) -> InputDecision:
    if not isinstance(transcript, str):
        return InputDecision(False, "", Route.STT_FAILURE, ReasonCode.TRANSCRIPT_INVALID_TYPE)
    normalized = " ".join(transcript.split()).strip()
    if not normalized:
        return InputDecision(False, "", Route.STT_FAILURE, ReasonCode.TRANSCRIPT_EMPTY)
    if len(normalized) > MAX_TRANSCRIPT_CHARS:
        return InputDecision(False, normalized, Route.STT_FAILURE, ReasonCode.TRANSCRIPT_TOO_LONG)
    tokens = [_clean_token(token) for token in normalized.split()]
    informative = [token for token in tokens if token and token not in _FILLERS]
    alphanumeric_chars = sum(character.isalnum() for token in informative for character in token)
    if not informative or (len(informative) == 1 and alphanumeric_chars < 4):
        return InputDecision(
            False,
            normalized,
            Route.STT_FAILURE,
            ReasonCode.TRANSCRIPT_LOW_INFORMATION,
        )
    return InputDecision(True, normalized)


def _clean_token(token: str) -> str:
    return "".join(
        character
        for character in token.casefold()
        if character.isalnum() or unicodedata.category(character).startswith("M")
    )


def route_input(transcript: str) -> InputDecision:
    folded = _policy_text(transcript)
    for reason, phrases in _UNSAFE_RULES:
        if any(_policy_text(phrase) in folded for phrase in phrases):
            return InputDecision(False, transcript, Route.UNSAFE, reason)
    for reason, phrases in _OFF_TOPIC_RULES:
        if any(_policy_text(phrase) in folded for phrase in phrases):
            return InputDecision(False, transcript, Route.OFF_TOPIC, reason)
    return InputDecision(True, transcript)


def _policy_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(
            character
            for character in token
            if character.isalnum() or unicodedata.category(character).startswith("M")
        )
        for token in normalized.split()
    )
