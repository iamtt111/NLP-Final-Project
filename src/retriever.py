# -*- coding: utf-8 -*-
"""BM25 + WAND retrieval engine."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .bm25_store import load_cache
from .tokenizer import tokenize


@dataclass
class ChunkHit:
    text: str
    source: str
    score: float


_cache: dict | None = None


def _ensure_cache() -> dict | None:
    global _cache
    if _cache is None:
        cache_path = Path("data/bm25_cache.pkl")
        _cache = load_cache(cache_path)
    return _cache


def reload_cache(cache_path: Path | None = None) -> None:
    global _cache
    path = cache_path or Path("data/bm25_cache.pkl")
    _cache = load_cache(path)


def get_candidate_chunk_indices(tokens: list[str], inverted_index: dict) -> list[int]:
    indices: set[int] = set()
    for token in tokens:
        for idx in inverted_index.get(token, []):
            indices.add(idx)
    return sorted(indices)


def _bm25_term_contribution(bm25, token: str, doc_index: int, query_weight: int = 1) -> float:
    frequencies = bm25.doc_freqs[doc_index]
    frequency = frequencies.get(token, 0)
    if frequency <= 0:
        return 0.0
    doc_length = bm25.doc_len[doc_index]
    avgdl = bm25.avgdl
    idf = bm25.idf.get(token, 0.0)
    k1 = bm25.k1
    b = bm25.b
    numerator = frequency * (k1 + 1)
    denominator = frequency + k1 * (1 - b + b * doc_length / avgdl)
    return float(idf * (numerator / denominator) * query_weight)


def _bm25_wand_topk(
    bm25,
    tokens: list[str],
    inverted_index: dict,
    top_k: int = 3,
    min_score: float = 0.5,
    candidate_indices: list[int] | None = None,
) -> list[tuple[int, float]]:
    token_counts = Counter(t for t in tokens if t)
    candidate_set = set(candidate_indices) if candidate_indices else None

    postings = []
    for token, qw in token_counts.items():
        docs = inverted_index.get(token) or []
        if not docs:
            continue
        if candidate_set is not None:
            docs = [d for d in docs if d in candidate_set]
            if not docs:
                continue
        max_score = 0.0
        for doc in docs:
            max_score = max(max_score, _bm25_term_contribution(bm25, token, doc, qw))
        postings.append({"token": token, "qw": qw, "docs": docs, "max_score": max_score, "pos": 0})
    postings.sort(key=lambda x: x["max_score"], reverse=True)

    if not postings:
        return []

    threshold = min_score
    heap: list[tuple[float, int]] = []

    while True:
        active = [p for p in postings if p["pos"] < len(p["docs"])]
        if not active:
            break
        active.sort(key=lambda x: x["docs"][x["pos"]])

        score_upper_bound = 0.0
        pivot_doc = None
        for pos, p in enumerate(active):
            score_upper_bound += p["max_score"]
            pivot_doc = p["docs"][p["pos"]]
            if score_upper_bound > threshold:
                break

        if pivot_doc is None or score_upper_bound <= threshold:
            break

        first_doc = active[0]["docs"][active[0]["pos"]]
        if first_doc == pivot_doc:
            score = 0.0
            for p in active:
                docs = p["docs"]
                pos_idx = p["pos"]
                if pos_idx < len(docs) and docs[pos_idx] == pivot_doc:
                    score += _bm25_term_contribution(bm25, p["token"], pivot_doc, p["qw"])
                    p["pos"] += 1
            if score >= min_score:
                if len(heap) < top_k:
                    heap.append((score, pivot_doc))
                    heap.sort()
                    threshold = max(min_score, heap[0][0]) if len(heap) >= top_k else min_score
                elif score > heap[0][0]:
                    heap[0] = (score, pivot_doc)
                    heap.sort()
                    threshold = max(min_score, heap[0][0])
        else:
            for p in active[:pos + 1]:
                docs = p["docs"]
                p["pos"] = bisect_left(docs, pivot_doc, p["pos"])

    return sorted([(doc_idx, s) for s, doc_idx in heap], key=lambda x: x[1], reverse=True)


def _decompose_query(question: str) -> list[str]:
    """Split compound questions by ？/? into sub-queries. Returns [question] if only one."""
    import re
    parts = re.split(r'[？?]', question)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) >= 2 else [question]


def _retrieve_single(
    question: str,
    bm25,
    docs: list,
    metas: list,
    inverted_index: dict,
    top_k: int,
    min_score: float,
) -> list[ChunkHit]:
    tokens = tokenize(question)
    if not tokens:
        return []
    candidates = get_candidate_chunk_indices(tokens, inverted_index)
    if not candidates:
        return []
    hits = _bm25_wand_topk(bm25, tokens, inverted_index, top_k, min_score, candidates)
    # If strict threshold returns nothing, fall back to best-effort (score ≥ 0) so the
    # LLM always receives some context instead of hitting the no_chunks short-circuit.
    if not hits:
        hits = _bm25_wand_topk(bm25, tokens, inverted_index, top_k, 0.0, candidates)
    return [
        ChunkHit(text=docs[idx], source=metas[idx].get("source", ""), score=score)
        for idx, score in hits
    ]


def retrieve_top_chunks_structured(
    question: str,
    top_k: int = 3,
    min_score: float = 0.5,
) -> list[tuple[str, list[ChunkHit]]]:
    """Returns [(sub_query, [ChunkHit, ...]), ...]. Single-query returns 1 entry."""
    cache = _ensure_cache()
    if not cache:
        return []

    bm25 = cache["bm25"]
    docs = cache["docs"]
    metas = cache["metas"]
    inverted_index = cache["inverted_index"]

    sub_queries = _decompose_query(question)
    sub_top_k = max(top_k, 8) if len(sub_queries) > 1 else top_k

    result = []
    seen_texts: set[str] = set()
    for sub_q in sub_queries:
        hits = _retrieve_single(sub_q, bm25, docs, metas, inverted_index, sub_top_k, min_score)
        unique_hits = []
        for h in hits:
            if h.text not in seen_texts:
                seen_texts.add(h.text)
                unique_hits.append(h)
        result.append((sub_q, unique_hits))
    return result


def retrieve_top_chunks(
    question: str,
    top_k: int = 3,
    min_score: float = 0.5,
) -> list[ChunkHit]:
    cache = _ensure_cache()
    if not cache:
        return []

    bm25 = cache["bm25"]
    docs = cache["docs"]
    metas = cache["metas"]
    inverted_index = cache["inverted_index"]

    sub_queries = _decompose_query(question)

    if len(sub_queries) == 1:
        return _retrieve_single(question, bm25, docs, metas, inverted_index, top_k, min_score)

    # Multi-part: retrieve with a wider top_k per sub-query to compensate for
    # vocabulary mismatch (relevant chunks may rank lower in a focused sub-query)
    sub_top_k = max(top_k, 8)
    seen_texts: set[str] = set()
    merged: list[ChunkHit] = []
    for sub_q in sub_queries:
        for hit in _retrieve_single(sub_q, bm25, docs, metas, inverted_index, sub_top_k, min_score):
            if hit.text not in seen_texts:
                seen_texts.add(hit.text)
                merged.append(hit)

    # Sort by score descending, keep top_k * len(sub_queries) to give LLM richer context
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged
