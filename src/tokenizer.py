# -*- coding: utf-8 -*-
"""Chinese-English hybrid tokenizer for BM25."""

from __future__ import annotations

import re

import jieba

jieba.setLogLevel(20)

ENG_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_STRIP_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def tokenize(text: str) -> list[str]:
    text = text or ""
    tokens: list[str] = []
    for seg in re.split(r"(\s+)", text):
        seg = seg.strip()
        if not seg:
            continue
        seg = _STRIP_PUNCT.sub("", seg)
        if not seg:
            continue
        if ENG_TOKEN.fullmatch(seg):
            tokens.append(seg.lower())
        else:
            tokens.extend(t for t in jieba.cut(seg) if t.strip())
    # Keep English single chars (formula variables), filter single CJK chars
    return [t for t in tokens if len(t) > 1 or ENG_TOKEN.fullmatch(t)]
