# -*- coding: utf-8 -*-
"""Sentence picker: fallback extractor using token overlap scoring."""

from __future__ import annotations

import re

from ._common import tokenize

ExtractorResult = tuple[str | None, float]


def pick_best_sentence(question: str, chunk: str) -> ExtractorResult:
    if not chunk:
        return None, 0.0

    q_tokens = set(tokenize(question))
    if not q_tokens:
        return None, 0.0

    sentences = _split_sentences(chunk)
    if not sentences:
        return None, 0.0

    scored = []
    for s in sentences:
        s_tokens = set(tokenize(s))
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens)
        length = len(s)
        length_penalty = 1.0
        if length < 10:
            length_penalty = 0.5
        elif length > 120:
            length_penalty = 0.6
        score = overlap * length_penalty
        scored.append((s, score, overlap))

    if not scored:
        return None, 0.0

    best, score, overlap = max(scored, key=lambda x: x[1])

    if overlap >= 3:
        conf = 0.7
    elif overlap == 2:
        conf = 0.55
    elif overlap == 1:
        conf = 0.4
    else:
        conf = 0.2

    return best.strip(), conf


def _split_sentences(chunk: str) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?])\s*", chunk)
    return [s for s in sentences if s.strip()]
