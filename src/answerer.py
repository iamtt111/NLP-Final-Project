# -*- coding: utf-8 -*-
"""
Answer pipeline: LLM-first, rules as acceleration.

Architecture:
  1. Cache hit     → instant return (0ms)
  2. Rule hit      → skip LLM call (only high-confidence factoid/definition)
  3. LLM primary   → Groq (main path for all questions)
  4. LLM fallback  → Gemini (if Groq fails)
  5. Sentence pick  → last resort if all LLMs fail
"""

from __future__ import annotations

import logging

from .extractors.factoid import extract_factoid
from .extractors.definition import extract_definition
from .post_process import post_process
from .cache import LLMCache
from .retriever import ChunkHit

log = logging.getLogger(__name__)

# Only use rules for very high-confidence, unambiguous cases
RULE_CONFIDENCE_THRESHOLD = 0.9


def _try_rules(question: str, chunk_texts: list[str]) -> tuple[str | None, str]:
    """Try rule extractors. Only return if confidence is very high."""
    # Factoid: dates/numbers are unambiguous when matched
    ans, conf = extract_factoid(question, chunk_texts)
    if ans and conf >= RULE_CONFIDENCE_THRESHOLD:
        return ans, "rule_factoid"

    # Definition: only trust exact pattern matches (conf=0.85 from template)
    ans, conf = extract_definition(question, chunk_texts)
    if ans and conf >= 0.85:
        return ans, "rule_definition"

    return None, ""


def _sort_chunks(hits: list[ChunkHit]) -> list[ChunkHit]:
    """Sort by BM25 score descending so the most relevant chunk is always first."""
    return sorted(hits, key=lambda h: h.score, reverse=True)


def _build_context(chunks: list[ChunkHit], structured: list[tuple[str, list[ChunkHit]]] | None) -> tuple[str, list[str]]:
    """Build context string and flat chunk_texts list."""
    if structured and len(structured) > 1:
        sections = []
        for sub_q, sub_hits in structured:
            sorted_hits = _sort_chunks(sub_hits)
            texts = [h.text for h in sorted_hits]
            if texts:
                sections.append(f"[{sub_q}]\n" + "\n---\n".join(texts))
        context = "\n\n".join(sections)
        all_texts = [h.text for _, hits in structured for h in _sort_chunks(hits)]
    else:
        sorted_chunks = _sort_chunks(chunks)
        all_texts = [c.text for c in sorted_chunks[:5]]
        context = "\n---\n".join(all_texts)
    return context, all_texts


def answer(
    question: str,
    chunks: list[ChunkHit],
    llm,
    cache: LLMCache | None,
    max_chars: int = 50,
    structured_chunks: list[tuple[str, list[ChunkHit]]] | None = None,
) -> tuple[str, str]:
    """Returns (answer, layer)."""
    if not chunks and not structured_chunks:
        return "NA", "no_chunks"

    context, chunk_texts = _build_context(chunks, structured_chunks)
    if not chunk_texts:
        return "NA", "no_chunks"

    # 1. Cache hit (instant)
    cache_key = None
    if cache is not None:
        cache_key = cache.hash_key(question, chunk_texts)
        if cached := cache.get(cache_key):
            return post_process(cached, question, max_chars=999), "cache"

    # 2. High-confidence rules (skip LLM for obvious factoids/definitions)
    rule_ans, rule_layer = _try_rules(question, chunk_texts)
    if rule_ans:
        return post_process(rule_ans, question, max_chars), rule_layer

    # 3. LLM (main path) — LLM output is already length-controlled by prompt,
    #    only strip prefixes, don't truncate
    if llm is not None:
        try:
            ans = llm.short_answer(question, context, max_chars)
            if cache is not None and cache_key:
                cache.set(cache_key, ans)
            return post_process(ans, question, max_chars=999), "llm"
        except Exception as e:
            log.warning(f"LLM failed: {e}")

    # 4. Fallback: pick best sentence from top chunk
    from .extractors.sentence_picker import pick_best_sentence
    picked, conf = pick_best_sentence(question, chunk_texts[0])
    if picked:
        return post_process(picked, question, max_chars), "sentence_picker"

    return "NA", "failed"
