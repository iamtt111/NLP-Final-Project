# -*- coding: utf-8 -*-
"""Streamlit UI for teacher review: select CSV, enter API keys, run QA, view results."""

from __future__ import annotations

import csv
import io
import os
import time
import threading
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


def read_csv_auto(path: Path) -> list[list[str]]:
    """Read CSV with automatic encoding detection."""
    raw = path.read_bytes()
    # Strip BOM if present
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for enc in ("utf-8", "cp950", "big5", "latin1"):
        try:
            text = raw.decode(enc)
            return [row for row in csv.reader(io.StringIO(text)) if row and row[0].strip()]
        except (UnicodeDecodeError, ValueError):
            continue
    return []

load_dotenv()

st.set_page_config(page_title="NLP QA 審閱系統", layout="wide")
st.title("NLP 課程 QA 審閱系統")

# ---------------------------------------------------------------------------
# 1. Settings section
# ---------------------------------------------------------------------------
st.header("設定")

col_csv, col_keys = st.columns([1, 1])

with col_csv:
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        st.warning("在 `input/` 目錄下沒有找到 CSV 檔案，請先放入測資。")
        st.stop()

    selected_csv = st.selectbox(
        "選擇測資 CSV",
        csv_files,
        format_func=lambda p: p.name,
    )

    # Preview
    if selected_csv:
        rows = read_csv_auto(selected_csv)
        st.caption(f"共 {len(rows)} 題")
        with st.expander("預覽前 5 題"):
            for i, row in enumerate(rows[:5], 1):
                st.write(f"{i}. {row[0]}")

with col_keys:
    groq_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="主要 LLM",
    )
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="備援 LLM",
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Execute section
# ---------------------------------------------------------------------------
if st.button("開始問答", type="primary", use_container_width=True):
    if not groq_key and not gemini_key:
        st.error("請至少輸入一個 API Key（Groq 或 Gemini）。")
        st.stop()

    # Read questions
    questions = [row[0] for row in read_csv_auto(selected_csv)]

    if not questions:
        st.error("CSV 檔案中沒有找到問題。")
        st.stop()

    # Initialize system
    from src.settings import load_settings
    from src.retriever import retrieve_top_chunks, reload_cache
    from src.answerer import answer
    from src.cache import LLMCache
    from src.llm_client import GroqClient, GeminiClient, LLMRouter

    import re

    settings = load_settings()

    with st.spinner("載入 BM25 索引..."):
        reload_cache()

    # Build LLM router with user-provided keys
    primary = None
    fallback = None
    timeout = settings.get("llm_timeout_seconds", 15.0)

    if groq_key:
        primary_model = settings.get("llm_primary", {}).get("model", "llama-3.3-70b-versatile")
        primary = GroqClient(api_key=groq_key, model=primary_model, timeout=timeout)
    if gemini_key:
        fallback_model = settings.get("llm_fallback", {}).get("model", "gemini-2.5-flash")
        fallback = GeminiClient(api_key=gemini_key, model=fallback_model, timeout=timeout)

    if not primary and fallback:
        primary = fallback
        fallback = None

    llm = LLMRouter(primary=primary, fallback=fallback) if primary else None

    cache_path = settings.get("cache_path", "data/llm_cache.json")
    llm_cache = LLMCache(cache_path)

    top_k = settings.get("top_k", 5)
    min_score = settings.get("bm25_min_score", 0.5)
    max_chars = settings.get("answer_max_chars", 200)

    # Run pipeline
    results = []
    progress = st.progress(0, text="執行中...")
    status_text = st.empty()

    for i, q in enumerate(questions):
        q_clean = re.sub(r"\s+", " ", q).strip()
        chunks = retrieve_top_chunks(q_clean, top_k=top_k, min_score=min_score)
        ans, layer = answer(q_clean, chunks, llm, llm_cache, max_chars=max_chars)
        results.append({"問題": q, "答案": ans, "Layer": layer})

        pct = (i + 1) / len(questions)
        progress.progress(pct, text=f"執行中... ({i+1}/{len(questions)})")
        status_text.caption(f"最新: {q[:40]}... → {ans[:40]}...")

    progress.progress(1.0, text="完成！")
    status_text.empty()

    st.session_state["results"] = results

# ---------------------------------------------------------------------------
# 3. Results section
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]

    st.divider()
    st.header("結果")

    # Summary
    layer_counts: dict[str, int] = {}
    for r in results:
        layer_counts[r["Layer"]] = layer_counts.get(r["Layer"], 0) + 1
    cols = st.columns(len(layer_counts) + 1)
    cols[0].metric("總題數", len(results))
    for i, (layer, count) in enumerate(sorted(layer_counts.items()), 1):
        cols[i].metric(layer, count)

    # Table
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=False,
        column_config={
            "問題": st.column_config.TextColumn("問題", width="large"),
            "答案": st.column_config.TextColumn("答案", width="large"),
            "Layer": st.column_config.TextColumn("Layer", width="small"),
        },
    )

    # Download
    output = io.StringIO()
    writer = csv.writer(output)
    for r in results:
        writer.writerow([r["問題"], r["答案"]])
    csv_bytes = output.getvalue().encode("utf-8-sig")

    st.download_button(
        label="下載結果 CSV",
        data=csv_bytes,
        file_name=f"answers_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
