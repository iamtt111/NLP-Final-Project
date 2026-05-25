# -*- coding: utf-8 -*-
"""Confidence gate: filter low-quality extractor results."""

from __future__ import annotations

import re

from ._common import tokenize

FACTOID_PATTERNS = [
    r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}",
    r"^\d+(?:\.\d+)?%?$",
    r"^\d+!$",
    r"^第\s*\d+\s*(?:章|節|頁)$",
]


def is_low_confidence(question: str, answer: str, source_chunk: str, conf: float) -> bool:
    if not answer or len(answer.strip()) < 2:
        return True
    if len(answer) > 100:
        return True
    if conf < 0.5:
        return True

    # High confidence from extractor: trust it
    if conf >= 0.7:
        return False

    # Factoid formats don't need token overlap
    if _is_factoid_format(answer):
        return False

    # Medium confidence: answer should have token overlap with question
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    if not (q_tokens & a_tokens):
        return True

    return False


def _is_factoid_format(answer: str) -> bool:
    return any(re.match(p, answer.strip()) for p in FACTOID_PATTERNS)
