# -*- coding: utf-8 -*-
"""BM25 JSONL store and pickle cache builder."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .tokenizer import tokenize

BM25_CACHE_VERSION = 1


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_cache(rows: list[dict], cache_path: Path) -> dict | None:
    if not rows:
        return None

    docs: list[str] = []
    metas: list[dict] = []
    tokenized: list[list[str]] = []
    inverted_index: dict[str, list[int]] = {}

    for i, row in enumerate(rows):
        text = row.get("text", "")
        source = row.get("source", "")
        tokens = row.get("tokens") or tokenize(text)

        docs.append(text)
        metas.append({"source": source, "text": text})
        tokenized.append(tokens)

        for token in set(tokens):
            inverted_index.setdefault(token, []).append(i)

    bm25 = BM25Okapi(tokenized)

    payload = {
        "version": BM25_CACHE_VERSION,
        "docs": docs,
        "metas": metas,
        "inverted_index": inverted_index,
        "bm25": bm25,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return payload


def load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("version") != BM25_CACHE_VERSION:
            return None
        return payload
    except Exception:
        return None
