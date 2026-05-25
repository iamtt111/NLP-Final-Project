# -*- coding: utf-8 -*-
"""Post-processing: remove prefixes, trim, truncate."""

from __future__ import annotations

import re

PREFIX_PATTERNS = [
    r"^答[:：]\s*",
    r"^答案[:：]\s*",
    r"^Answer[:：]\s*",
    r"^The answer is\s+",
]


def post_process(ans: str, question: str, max_chars: int = 50) -> str:
    ans = ans.strip()
    for p in PREFIX_PATTERNS:
        ans = re.sub(p, "", ans, flags=re.IGNORECASE)
    ans = ans.strip(' 。"\'""''')
    if len(ans) > max_chars:
        ans = ans[:max_chars].rstrip()
    return ans
