# -*- coding: utf-8 -*-
"""Enumeration extractor: list/steps questions."""

from __future__ import annotations

import re

from ._common import strip_question_words

ExtractorResult = tuple[str | None, float]

ENUM_TRIGGERS_ZH = (
    "有哪些", "包含哪些", "有什麼", "包括什麼",
    "列出", "舉例", "舉出", "請說明", "種類",
    "步驟", "流程", "順序", "階段",
)

ENUM_TRIGGERS_EN = (
    "list", "what are the", "name the", "enumerate",
    "types of", "kinds of", "categories of", "examples of",
    "steps", "stages", "phases",
)

LIST_ITEM_PATTERNS = [
    r"^\s*\(?\d+[\.\)）]\s*(.+)$",
    r"^\s*[①-⑳]\s*(.+)$",
    r"^\s*\(?[a-zA-Z][\.\)）]\s*(.+)$",
    r"^\s*[\-•·▪▫◦‣⁃]\s*(.+)$",
]

INLINE_LIST_PATTERN = r"(?:包含|包括|有|分為|分成)[:：]?\s*(.+?)[。\n]"


def extract_enumeration(question: str, chunks: list[str]) -> ExtractorResult:
    q_lower = question.lower()
    triggers = ENUM_TRIGGERS_ZH + ENUM_TRIGGERS_EN
    if not any(t in q_lower or t in question for t in triggers):
        return None, 0.0

    target = strip_question_words(question)

    for chunk in chunks:
        if target and target.lower() not in chunk.lower():
            continue
        items = _find_list_items(chunk)
        if len(items) >= 2:
            ans = "、".join(items[:6])
            if len(ans) <= 100:
                return ans, 0.85
            return ans[:100].rstrip("、") + "...", 0.7

    # Fallback: inline list
    for chunk in chunks:
        if m := re.search(INLINE_LIST_PATTERN, chunk):
            ans = m.group(1).strip()
            if 5 <= len(ans) <= 100:
                return ans, 0.7

    return None, 0.0


def _find_list_items(chunk: str) -> list[str]:
    lines = chunk.split("\n")
    items = []
    for line in lines:
        for pat in LIST_ITEM_PATTERNS:
            if m := re.match(pat, line):
                items.append(m.group(1).strip())
                break
    return items
