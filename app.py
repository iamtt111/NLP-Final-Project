# -*- coding: utf-8 -*-
"""Streamlit UI for teacher review: select CSV, enter API keys, run QA, view results."""

from __future__ import annotations

import csv
import io
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


def parse_keys(raw: str) -> list[str]:
    """Split a multi-line / comma-separated key string into a clean list."""
    keys = []
    for part in raw.replace(",", "\n").splitlines():
        k = part.strip()
        if k:
            keys.append(k)
    return keys


def env_keys(prefix: str) -> str:
    """Collect API keys from env: PREFIX, PREFIX_1 … PREFIX_4 (deduplicated, order preserved)."""
    seen: set[str] = set()
    result: list[str] = []
    for var in [prefix] + [f"{prefix}_{i}" for i in range(1, 5)]:
        k = os.environ.get(var, "").strip()
        if k and k not in seen:
            seen.add(k)
            result.append(k)
    return "\n".join(result)


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
    groq_keys_input = st.text_area(
        "Groq API Keys（每行一把，最多 4 把）",
        value=env_keys("GROQ_API_KEY"),
        height=120,
        help="主要 LLM；多把 key 平行分流",
    )
    gemini_keys_input = st.text_area(
        "Gemini API Keys（每行一把，最多 4 把）",
        value=env_keys("GEMINI_API_KEY"),
        height=80,
        help="備援 LLM（Groq 全部失敗時啟用）",
    )

groq_keys = parse_keys(groq_keys_input)
gemini_keys = parse_keys(gemini_keys_input)

if groq_keys or gemini_keys:
    total = len(groq_keys) + len(gemini_keys)
    st.caption(f"已偵測到 {len(groq_keys)} 把 Groq key、{len(gemini_keys)} 把 Gemini key，共 {total} 把")

st.divider()

# ---------------------------------------------------------------------------
# 2. Execute section
# ---------------------------------------------------------------------------
if st.button("開始問答", type="primary", use_container_width=True):
    if not groq_keys and not gemini_keys:
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

    # Build LLM router with key pools
    timeout = settings.get("llm_timeout_seconds", 15.0)
    primary_model = settings.get("llm_primary", {}).get("model", "llama-3.3-70b-versatile")
    fallback_model = settings.get("llm_fallback", {}).get("model", "gemini-2.5-flash")

    groq_clients = [GroqClient(api_key=k, model=primary_model, timeout=timeout) for k in groq_keys]
    gemini_clients = [GeminiClient(api_key=k, model=fallback_model, timeout=timeout) for k in gemini_keys]

    # Pair each Groq key with a Gemini key (cycle Gemini if fewer keys than Groq)
    n_pairs = len(groq_clients) or len(gemini_clients)
    routers = []
    for i in range(n_pairs):
        g   = groq_clients[i]                              if i < len(groq_clients)   else None
        gem = gemini_clients[i % len(gemini_clients)]      if gemini_clients          else None
        primary = g or gem
        fallback = gem if g else None
        routers.append(LLMRouter(primary=primary, fallback=fallback))

    n_workers = min(len(routers), len(questions))

    cache_path = settings.get("cache_path", "data/llm_cache.json")
    llm_cache = LLMCache(cache_path)

    top_k = settings.get("top_k", 5)
    min_score = settings.get("bm25_min_score", 0.5)
    max_chars = settings.get("answer_max_chars", 200)

    # Run pipeline — parallel across key pool
    results = [None] * len(questions)
    progress = st.progress(0, text=f"平行執行中（{n_workers} workers）...")
    status_text = st.empty()

    def _run_one(item):
        i, q_raw = item
        q = re.sub(r"\s+", " ", q_raw).strip()
        chunks = retrieve_top_chunks(q, top_k=top_k, min_score=min_score)
        ans, layer = "NA", "failed"
        for offset in range(len(routers)):
            router = routers[(i + offset) % len(routers)]
            ans, layer = answer(q, chunks, router, llm_cache, max_chars=max_chars)
            if layer not in ("sentence_picker", "failed"):
                break
        return i, q_raw, ans, layer

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_one, (i, q)): i for i, q in enumerate(questions)}
        completed = 0
        for future in as_completed(futures):
            i, q_raw, ans, layer = future.result()
            results[i] = {"問題": q_raw, "答案": ans, "Layer": layer}
            completed += 1
            progress.progress(completed / len(questions), text=f"平行執行中... ({completed}/{len(questions)})")
            status_text.caption(f"最新完成: {q_raw[:40]}...")

    progress.progress(1.0, text="完成！")
    status_text.empty()

    # Build CSV bytes and save to output/ exactly once
    out_buf = io.StringIO()
    out_writer = csv.writer(out_buf)
    for r in results:
        out_writer.writerow([r["問題"], r["答案"]])
    csv_bytes = out_buf.getvalue().encode("utf-8-sig")

    input_stem = Path(selected_csv).stem
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_filename = f"{input_stem}_{timestamp}.csv"

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    (out_dir / out_filename).write_bytes(csv_bytes)

    st.session_state["results"] = results
    st.session_state["csv_bytes"] = csv_bytes
    st.session_state["out_filename"] = out_filename

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
    out_filename = st.session_state["out_filename"]
    csv_bytes = st.session_state["csv_bytes"]

    st.download_button(
        label="下載結果 CSV",
        data=csv_bytes,
        file_name=out_filename,
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"已自動儲存至 output/{out_filename}")
