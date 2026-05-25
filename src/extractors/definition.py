# -*- coding: utf-8 -*-
"""Definition extractor: 'X is Y' / 'X 是 Y' pattern matching."""

from __future__ import annotations

import re

from ._common import strip_question_words, is_chinese

ExtractorResult = tuple[str | None, float]

DEF_PATTERNS_ZH = [
    r"{target}\s*(?:是|即|指|表示|定義為|意指|代表)\s*(.+?)[。\n；;]",
    r"{target}\s*[:：]\s*(.+?)[。\n；;]",
]

DEF_PATTERNS_EN = [
    r"{target}\s+(?:is|are|refers\s+to|means|denotes|is\s+defined\s+as|is\s+a\s+kind\s+of)\s+(.+?)[.\n;]",
    r"{target}\s*[:\-—]\s+(.+?)[.\n;]",
]


def extract_definition(question: str, chunks: list[str]) -> ExtractorResult:
    target = strip_question_words(question)
    if not target or len(target) < 2:
        return None, 0.0

    target_re = re.escape(target)

    patterns = []
    if is_chinese(question):
        patterns += [p.replace("{target}", target_re) for p in DEF_PATTERNS_ZH]
    patterns += [p.replace("{target}", target_re) for p in DEF_PATTERNS_EN]

    for chunk in chunks:
        for pat in patterns:
            if m := re.search(pat, chunk, re.IGNORECASE):
                ans = m.group(1).strip()
                ans = _trim_to_one_clause(ans)
                if 2 <= len(ans) <= 100:
                    return ans, 0.85

    # Fallback: find sentence containing target
    for chunk in chunks:
        if target.lower() in chunk.lower():
            sent = _find_sentence_with(target, chunk)
            if sent and 5 <= len(sent) <= 150:
                return sent, 0.5

    return None, 0.0


def _trim_to_one_clause(ans: str) -> str:
    for sep in ("，", ",", "；", ";"):
        if sep in ans:
            head = ans.split(sep, 1)[0].strip()
            if len(head) >= 5:
                return head
    return ans


def _find_sentence_with(target: str, chunk: str) -> str | None:
    sentences = re.split(r"(?<=[。！？.!?\n])", chunk)
    for s in sentences:
        if target.lower() in s.lower():
            return s.strip()
    return None
