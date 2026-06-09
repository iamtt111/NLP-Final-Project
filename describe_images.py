# -*- coding: utf-8 -*-
"""
Standalone script: extract images from PDF slides and generate text descriptions.

Usage:
    python describe_images.py [--corpus ./corpus]

Output:
    For each PDF with images, creates a corresponding .txt file in corpus/:
        c01_intro.pdf  →  c01_intro_images.txt

    These .txt files are picked up by build_corpus.py automatically.
    Re-running only processes new/changed images (cached in data/image_descriptions.json).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate text descriptions for PDF slide images")
    parser.add_argument("--corpus", default="./corpus", help="Corpus folder path")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    pdf_files = sorted(corpus.rglob("*.pdf"))

    if not pdf_files:
        print(f"[error] No PDF files found in {corpus}")
        return

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[error] GROQ_API_KEY not set in .env")
        return

    from src.image_describer import describe_pdf_images

    print(f"[info] Found {len(pdf_files)} PDF files (using Groq Vision)")
    started = time.perf_counter()
    total_images = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        output_dir = Path(args.corpus) / "slides" / "image_description"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}_images.txt"

        # Skip if description file exists and is newer than the PDF
        if output_path.exists() and output_path.stat().st_mtime >= pdf_path.stat().st_mtime:
            print(f"  [{i}/{len(pdf_files)}] SKIP (up to date): {pdf_path.name}")
            continue

        print(f"  [{i}/{len(pdf_files)}] Processing: {pdf_path.name}...")
        page_descs = describe_pdf_images(pdf_path, api_key=api_key)

        if not page_descs:
            print(f"    No images found")
            continue

        # Write descriptions as a text file
        lines = [f"Image descriptions from: {pdf_path.name}", ""]
        img_count = 0
        for page_num in sorted(page_descs.keys()):
            for desc in page_descs[page_num]:
                img_count += 1
                lines.append(f"[Page {page_num + 1}, Image {img_count}] {desc}")
                lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        total_images += img_count
        print(f"    {img_count} images → {output_path.name}")

    elapsed = time.perf_counter() - started
    print(f"[done] {total_images} image descriptions in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
