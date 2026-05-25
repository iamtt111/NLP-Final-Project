# -*- coding: utf-8 -*-
"""Common utilities for extractors."""

from __future__ import annotations

import re

from ..tokenizer import tokenize  # noqa: F401

ENG_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
CJK = re.compile(r"[一-鿿]")

QUESTION_MARKS = ("?", "？")

ZH_QUESTION_WORDS = (
    "什麼是", "什麼叫", "什麼叫做", "什麼意思", "是什麼", "什麼",
    "如何", "怎麼", "為什麼", "為何",
    "幾月幾號", "幾月幾日", "什麼時候", "哪時", "何時",
    "多少", "幾", "哪個", "哪些", "哪裡",
)

EN_QUESTION_WORDS = (
    "what is", "what are", "what's",
    "define", "definition of",
    "how", "why", "when", "where", "which",
    "list", "name", "give",
)


def is_chinese(text: str) -> bool:
    return bool(CJK.search(text))


def strip_question_words(question: str) -> str:
    q = question.strip()
    for mark in QUESTION_MARKS:
        q = q.rstrip(mark).strip()
    q_lower = q.lower()
    for w in EN_QUESTION_WORDS:
        if q_lower.startswith(w + " "):
            return q[len(w):].strip()
        if q_lower.startswith(w) and len(q_lower) == len(w):
            return ""
    for w in ZH_QUESTION_WORDS:
        if q.startswith(w):
            return q[len(w):].strip()
        if q.endswith(w):
            return q[:-len(w)].strip()
    return q
