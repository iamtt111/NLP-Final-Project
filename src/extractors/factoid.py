# -*- coding: utf-8 -*-
"""Factoid extractor: dates, numbers, chapters, formulas, person names."""

from __future__ import annotations

import re

ExtractorResult = tuple[str | None, float]

DATE_TRIGGERS = (
    "幾月幾號", "幾月幾日", "什麼時候", "哪天", "何時", "日期", "幾年",
    "when", "what date", "what day", "which date",
)

NUMBER_TRIGGERS = (
    "多少", "幾個", "幾筆", "幾次", "多大", "多長", "多遠", "百分之幾",
    "how many", "how much", "what percentage", "value of",
)

CHAPTER_TRIGGERS = (
    "第幾章", "第幾節", "哪一章", "哪一節", "第幾頁",
    "chapter", "section", "page",
)

FORMULA_TRIGGERS = (
    "公式", "方程式", "算式", "formula", "equation",
)

PERSON_TRIGGERS = (
    "誰", "誰是", "誰提出", "誰發明",
    "who", "who is", "who proposed", "who invented",
)

DATE_PATTERNS = [
    r"\d{4}[/\-.年]\d{1,2}[/\-.月]\d{1,2}(?:日)?",
    r"\d{1,2}月\d{1,2}日",
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{1,2}(?:,?\s*\d{4})?",
]

NUMBER_PATTERNS = [
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+!\b",
    r"\b\d+(?:\.\d+)?\b",
]

CHAPTER_PATTERNS = [
    r"(?:第|chapter\s*|ch\.?\s*)\s*\d+\s*(?:章|節|課)?",
    r"(?:第|page\s*|p\.?\s*)\s*\d+\s*頁?",
]

FORMULA_PATTERNS = [
    r"[A-Za-z_]+\s*=\s*[^\n。.]{1,80}",
]


def _hit_trigger(question: str, q_lower: str, triggers: tuple[str, ...]) -> bool:
    for t in triggers:
        if t in q_lower or t in question:
            return True
    return False


EVENT_WORDS = [
    "期末考", "期中考", "midterm", "final exam", "final", "deadline",
    "報告", "繳交", "上課", "放假", "停課", "補課", "考試",
]


def _extract_event_words(question: str) -> list[str]:
    """Extract event keywords from question for date context matching."""
    q = question.lower()
    return [w for w in EVENT_WORDS if w.lower() in q]


def extract_factoid(question: str, chunks: list[str]) -> ExtractorResult:
    q_lower = question.lower()

    if _hit_trigger(question, q_lower, DATE_TRIGGERS):
        # Extract event keywords from question (e.g. "期末考", "midterm")
        event_words = _extract_event_words(question)
        for chunk in chunks:
            # If question has event words, find date near that event
            if event_words:
                for ew in event_words:
                    pos = chunk.lower().find(ew.lower())
                    if pos < 0:
                        continue
                    # Only look AFTER the event word to avoid grabbing a previous date
                    after = chunk[pos:pos + 80]
                    for pat in DATE_PATTERNS:
                        if m := re.search(pat, after):
                            return m.group(0).strip(), 0.95
            else:
                # No event word — take first date found
                for pat in DATE_PATTERNS:
                    if m := re.search(pat, chunk):
                        return m.group(0).strip(), 0.9

    if _hit_trigger(question, q_lower, NUMBER_TRIGGERS):
        for chunk in chunks:
            for pat in NUMBER_PATTERNS:
                if m := re.search(pat, chunk):
                    return m.group(0).strip(), 0.85

    if _hit_trigger(question, q_lower, CHAPTER_TRIGGERS):
        for chunk in chunks:
            for pat in CHAPTER_PATTERNS:
                if m := re.search(pat, chunk, re.IGNORECASE):
                    return m.group(0).strip(), 0.9

    if _hit_trigger(question, q_lower, FORMULA_TRIGGERS):
        for chunk in chunks:
            for pat in FORMULA_PATTERNS:
                if m := re.search(pat, chunk):
                    return m.group(0).strip(), 0.8

    if _hit_trigger(question, q_lower, PERSON_TRIGGERS):
        for chunk in chunks:
            if m := re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", chunk):
                return m.group(0), 0.6
            if m := re.search(r"(?:由|是|為|提出者?是?)\s*([一-鿿]{2,4})\s*(?:提出|發明|定義)", chunk):
                return m.group(1), 0.65

    return None, 0.0
