# -*- coding: utf-8 -*-
"""One-time corpus indexing: parse documents -> chunk -> tokenize -> build BM25 index."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.parser import parse_file
from src.chunker import chunk_text
from src.tokenizer import tokenize
from src.bm25_store import write_jsonl, build_cache
from src.settings import load_settings


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from corpus")
    parser.add_argument("--corpus", default=None, help="Corpus folder path")
    args = parser.parse_args()

    settings = load_settings()
    corpus_folder = Path(args.corpus) if args.corpus else Path(settings["corpus_folder"])
    chunk_size = settings.get("chunk_size", 300)
    chunk_overlap = settings.get("chunk_overlap", 60)

    jsonl_path = Path("data/bm25_chunks.jsonl")
    cache_path = Path("data/bm25_cache.pkl")

    extensions = {".pdf", ".pptx", ".docx", ".txt", ".srt", ".vtt"}
    files = [f for f in corpus_folder.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print(f"[error] No files found in {corpus_folder}")
        return

    print(f"[info] Found {len(files)} files in {corpus_folder}")
    started = time.perf_counter()

    rows: list[dict] = []
    for i, file_path in enumerate(files, 1):
        text = parse_file(file_path)
        if not text.strip():
            print(f"  [{i}/{len(files)}] SKIP (empty): {file_path.name}")
            continue

        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
        for chunk in chunks:
            tokens = tokenize(chunk)
            rows.append({
                "source": file_path.name,
                "text": chunk,
                "tokens": tokens,
            })
        print(f"  [{i}/{len(files)}] OK: {file_path.name} -> {len(chunks)} chunks")

    write_jsonl(rows, jsonl_path)
    print(f"[info] Written {len(rows)} chunks to {jsonl_path}")

    build_cache(rows, cache_path)
    elapsed = time.perf_counter() - started
    print(f"[done] Index built in {elapsed:.2f}s | {len(rows)} chunks from {len(files)} files")


if __name__ == "__main__":
    main()
