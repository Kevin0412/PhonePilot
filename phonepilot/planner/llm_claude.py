"""Claude / Anthropic 文本 LLM 适配器（Messages API）。

Anthropic 无原生 json 模式：靠 system 提示词强约束「只输出 JSON」+ extract_json 兜底。
"""

from __future__ import annotations

import os
from typing import Optional

from .llm_base import LLMClient, extract_json, http_post_json


class ClaudeClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 30.0,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("缺少 ANTHROPIC_API_KEY（放入 .env 或环境变量）")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.version = anthropic_version

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }
        resp = http_post_json(self.base_url, "/v1/messages", headers, payload, self.timeout)
        # content 是块列表，取首个 text 块
        text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                break
        return extract_json(text)
