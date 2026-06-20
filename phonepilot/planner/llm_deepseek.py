"""DeepSeek 文本 LLM 适配器（OpenAI 兼容 /chat/completions）。

支持「快速路径」关思考：thinking=False 时附 ``thinking={"type":"disabled"}``（不支持的
模型会忽略该字段）。response_format=json_object + 提示词中的「JSON」要求双保险输出 JSON。
"""

from __future__ import annotations

import os
from typing import Optional

from .llm_base import LLMClient, extract_json, http_post_json


class DeepSeekClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 8.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY（放入 .env 或环境变量）")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if not thinking:
            payload["thinking"] = {"type": "disabled"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = http_post_json(self.base_url, "/chat/completions", headers, payload, self.timeout)
        content = resp["choices"][0]["message"]["content"]
        return extract_json(content)
