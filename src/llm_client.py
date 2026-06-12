# -*- coding: utf-8 -*-
"""LLM clients: Groq (primary) + Gemini (fallback) with auto-routing."""

from __future__ import annotations

import logging
import time
import threading
from typing import Protocol

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一個精確的文件問答助手，專門從給定文件中擷取答案。

規則：
1. 答案必須來自文件內容，不可自行編造
2. 只輸出答案本身，禁止任何前綴（「答：」「答案是」「The answer is」「根據文件」）
3. 答案要完整且切中要點：
   - 日期/數值題 → 輸出該日期或數值（若文件有明確年份/數字，必須列出）
   - 人名題 → 輸出人名
   - 列舉題 → 列出所有相關項目，用頓號分隔（例：A、B、C）
   - 列舉「挑戰」或「困難」題 → 只列困難類型（如 sarcasm、negation、ambiguity），不可將評估指標（Precision、Recall、F1、Accuracy）列為挑戰
   - 定義/解釋題 → 用1~2句話完整說明；若描述某參數「高/低」或「大/小」的效果，必須準確對應，不可混淆兩個方向的效果
   - 其他題型 → 根據文件內容給出完整回答，不超過 {max_chars} 字
4. 寧可多給一點資訊也不要遺漏重點
5. 答案語言與題目一致（中文題回繁體中文、英文題回英文），但專有名詞保留原文
6. 若文件中找不到答案，輸出 NA，查無答案
7. 若問題含多個子問題（以「？」分隔），必須依序回答每個子問題，不可遺漏任何一個"""

USER_TEMPLATE = """\
題目：{question}

文件內容：
{context}"""


class LLMClient(Protocol):
    def short_answer(self, question: str, context: str, max_chars: int = 50) -> str: ...


class GroqClient:
    """Primary LLM. Free tier: 30 RPM, 14400 RPD. LPU inference is fast."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: float = 10.0, max_tokens: int = 1024):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def short_answer(self, question: str, context: str, max_chars: int = 50) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(max_chars=max_chars)},
                {"role": "user", "content": USER_TEMPLATE.format(
                    question=question, context=context)},
            ],
            temperature=0,
            top_p=1,
            max_tokens=self.max_tokens,
            seed=42,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()


class _RateLimiter:
    """Simple token-bucket rate limiter (thread-safe)."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm  # seconds between requests
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_until = self._last + self._interval
            # Claim the next slot and release the lock immediately
            self._last = max(now, wait_until)
            sleep_time = max(0.0, wait_until - now)
        time.sleep(sleep_time)  # sleep outside lock so workers run concurrently


class GeminiClient:
    """Fallback LLM. Uses AI Studio key (no Vertex/GCP needed)."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", timeout: float = 15.0, max_tokens: int = 1024):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._limiter = _RateLimiter(rpm=10)  # Gemini 2.5 Flash free tier: 15 RPM

    def short_answer(self, question: str, context: str, max_chars: int = 50) -> str:
        from google.genai import types
        import re

        prompt = (
            SYSTEM_PROMPT.format(max_chars=max_chars) + "\n\n"
            + USER_TEMPLATE.format(question=question, context=context)
        )

        for attempt in range(3):
            self._limiter.wait()
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        top_p=1,
                        top_k=1,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                return resp.text.strip()
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 2:
                    match = re.search(r"retryDelay[\"'\s:]+(\d+)s", err_str)
                    delay = int(match.group(1)) if match else 30
                    if delay <= 60:
                        log.warning(f"GeminiClient: rate limited, waiting {delay}s (attempt {attempt + 1}/3)")
                        time.sleep(delay)
                        continue
                raise


class LLMRouter:
    """Primary-fallback auto-switch with unified error handling."""

    def __init__(self, primary: LLMClient, fallback: LLMClient | None = None):
        self.primary = primary
        self.fallback = fallback

    def short_answer(self, question: str, context: str, max_chars: int = 50) -> str:
        try:
            return self.primary.short_answer(question, context, max_chars)
        except Exception as e:
            log.warning(f"primary LLM failed: {type(e).__name__}: {e}")
            if self.fallback:
                return self.fallback.short_answer(question, context, max_chars)
            raise


class KeyPoolClient:
    """Round-robin pool — distributes requests across multiple LLM clients (thread-safe)."""

    def __init__(self, clients: list[LLMClient]):
        if not clients:
            raise ValueError("KeyPoolClient requires at least one client")
        self._clients = clients
        self._idx = 0
        self._lock = threading.Lock()

    def _next(self) -> LLMClient:
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client

    def short_answer(self, question: str, context: str, max_chars: int = 50) -> str:
        last_err: Exception | None = None
        for _ in range(len(self._clients)):
            try:
                return self._next().short_answer(question, context, max_chars)
            except Exception as e:
                log.warning(f"KeyPoolClient: key failed ({type(e).__name__}), trying next")
                last_err = e
        raise last_err
