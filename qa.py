# -*- coding: utf-8 -*-
"""Main QA pipeline: batch CSV -> CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from src.settings import load_settings
from src.retriever import retrieve_top_chunks, reload_cache
from src.answerer import answer
from src.cache import LLMCache
from src.llm_client import GroqClient, GeminiClient, LLMRouter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_llm: LLMRouter | None = None
_cache: LLMCache | None = None
_settings: dict = {}


def _init_llm(settings: dict) -> LLMRouter:
    primary_cfg = settings.get("llm_primary", {})
    fallback_cfg = settings.get("llm_fallback", {})
    timeout = settings.get("llm_timeout_seconds", 10.0)

    primary = None
    fallback = None

    max_tokens = settings.get("llm_max_tokens", 1024)

    groq_key = os.environ.get(primary_cfg.get("api_key_env", "GROQ_API_KEY"), "")
    if groq_key:
        primary = GroqClient(api_key=groq_key, model=primary_cfg.get("model", "qwen-2.5-32b"), timeout=timeout, max_tokens=max_tokens)

    gemini_key = os.environ.get(fallback_cfg.get("api_key_env", "GEMINI_API_KEY"), "")
    if gemini_key:
        fallback = GeminiClient(api_key=gemini_key, model=fallback_cfg.get("model", "gemini-2.5-flash"), timeout=timeout, max_tokens=max_tokens)

    if not primary and fallback:
        primary = fallback
        fallback = None
    if not primary:
        return None

    return LLMRouter(primary=primary, fallback=fallback)


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950", "gb2312", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"


def read_csv_column_a(path: str) -> list[str]:
    enc = detect_encoding(path)
    with open(path, "r", encoding=enc, newline="") as f:
        return [row[0] for row in csv.reader(f) if row and row[0].strip()]


def write_csv(path: str, pairs: list[tuple[str, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Add timestamp to filename: answers.csv → answers_20260525_143012.csv
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stamped = p.parent / f"{p.stem}_{stamp}{p.suffix}"
    try:
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(pairs)
        log.info(f"Saved to {p}")
    except PermissionError:
        log.warning(f"Cannot write to {p} (file locked?), saving to {stamped}")
        with open(stamped, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(pairs)
        log.info(f"Saved to {stamped}")
        return
    # Also save timestamped copy
    with open(stamped, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(pairs)
    log.info(f"Copy saved to {stamped}")


def answer_pipeline(question: str) -> tuple[str, str]:
    global _llm, _cache, _settings
    q = normalize_query(question)
    top_k = _settings.get("top_k", 3)
    min_score = _settings.get("bm25_min_score", 0.5)
    max_chars = _settings.get("answer_max_chars", 50)

    chunks = retrieve_top_chunks(q, top_k=top_k, min_score=min_score)
    ans, layer = answer(q, chunks, _llm, _cache, max_chars=max_chars)
    return ans, layer


def main():
    global _llm, _cache, _settings

    parser = argparse.ArgumentParser(description="Batch QA: CSV -> CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--workers", type=int, default=None, help="Thread pool workers")
    args = parser.parse_args()

    _settings = load_settings()
    workers = args.workers or _settings.get("workers", 4)

    log.info("Loading BM25 index...")
    reload_cache()

    log.info("Initializing LLM...")
    _llm = _init_llm(_settings)
    if _llm is None:
        log.warning("No LLM API key found. Running in rules-only mode (no LLM fallback).")

    cache_path = _settings.get("cache_path", "data/llm_cache.json")
    _cache = LLMCache(cache_path)

    questions = read_csv_column_a(args.input)
    log.info(f"Loaded {len(questions)} questions from {args.input}")

    started = time.perf_counter()
    layer_counts: dict[str, int] = {}

    done_count = 0
    total = len(questions)
    count_lock = threading.Lock()

    def process(q: str) -> tuple[str, str, str]:
        nonlocal done_count
        ans, layer = answer_pipeline(q)
        with count_lock:
            done_count += 1
            log.info(f"[{done_count}/{total}] ({layer}) {q[:40]} → {ans[:60]}")
        return q, ans, layer

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process, questions))

    pairs = []
    for q, ans, layer in results:
        pairs.append((q, ans))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    elapsed = time.perf_counter() - started
    write_csv(args.output, pairs)

    log.info(f"[done] {len(pairs)} questions / {elapsed:.2f}s / {elapsed/max(len(pairs),1)*1000:.1f}ms per question")
    log.info(f"Layer hits: {layer_counts}")


if __name__ == "__main__":
    main()
