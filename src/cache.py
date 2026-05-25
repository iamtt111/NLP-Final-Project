# -*- coding: utf-8 -*-
"""LLM result cache: avoid duplicate API calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LLMCache:
    def __init__(self, path: str | Path = "data/llm_cache.json"):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self._data = loaded if isinstance(loaded, dict) else {}
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def hash_key(question: str, chunks: list[str]) -> str:
        content = question + "|||" + "|||".join(chunks)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._save()
