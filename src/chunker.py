# -*- coding: utf-8 -*-
"""Text chunking with sentence-boundary awareness and overlap."""

from __future__ import annotations

import re


def chunk_text(text: str, size: int = 300, overlap: int = 60) -> list[str]:
    raw = re.split(r"(?<=[。！？.!?\n])", text or "")
    sentences = [s for s in raw if s.strip()]
    if not sentences:
        stripped = (text or "").strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    index = 0
    while index < len(sentences):
        char_count = 0
        next_index = index
        while next_index < len(sentences):
            char_count += len(sentences[next_index])
            next_index += 1
            if char_count >= size:
                break
        if next_index == index:
            next_index = index + 1
        chunks.append("".join(sentences[index:next_index]))

        overlap_chars = 0
        overlap_start = next_index
        for pointer in range(next_index - 1, index, -1):
            overlap_chars += len(sentences[pointer])
            if overlap_chars >= overlap:
                overlap_start = pointer
                break
        index = overlap_start if overlap_start > index else next_index

    return chunks
