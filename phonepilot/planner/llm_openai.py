"""通用 OpenAI 兼容文本 LLM 适配器。

很多国产模型(MiMo / Qwen / Kimi / GLM)都提供 OpenAI 兼容的 /chat/completions。
用一个客户端经 provider 预置或显式 base_url/model 切换，无需为每家单写适配器。
DeepSeek 因有专属字段(json_object/thinking)单独保留 DeepSeekClient。

默认不发 response_format(部分端点不支持)，仅靠提示词「只输出 JSON」+ extract_json 兜底；
需要时可置 json_mode=True。
"""

from __future__ import annotations

import os
from typing import Optional

from .llm_base import LLMClient, extract_json, http_post_json

# provider -> (base_url, default_model, env_key)
TEXT_PRESETS: dict[str, tuple[str, str, str]] = {
    "mimo": ("https://token-plan-cn.xiaomimimo.com/v1", "mimo-v2.5", "MIMO_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "DASHSCOPE_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "moonshot-v1-8k", "MOONSHOT_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", "GLM_API_KEY"),
}


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        *,
        provider: str = "mimo",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        env_key: Optional[str] = None,
        timeout: float = 30.0,
        json_mode: bool = False,
    ) -> None:
        pu, pm, pe = TEXT_PRESETS.get(provider, (None, "", ""))
        self.provider = provider
        self.base_url = base_url or pu
        self.model = model or pm
        resolved_env = env_key or pe
        self.api_key = api_key or (os.environ.get(resolved_env) if resolved_env else None)
        self.timeout = timeout
        self.json_mode = json_mode
        if not self.base_url:
            raise RuntimeError(f"OpenAI 兼容 provider={provider} 缺少 base_url（配置 llm.openai.base_url）")
        if not self.model:
            raise RuntimeError(f"OpenAI 兼容 provider={provider} 缺少 model")
        if not self.api_key:
            raise RuntimeError(f"缺少 API key（环境变量 {resolved_env or '未知'}）")

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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = http_post_json(self.base_url, "/chat/completions", headers, payload, self.timeout)
        msg = resp["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, list):  # 个别实现返回内容块列表
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        # 推理模型(如 MiMo)有时把答案留在 reasoning_content；content 为空时回退
        if not (content and str(content).strip()):
            content = msg.get("reasoning_content") or ""
        return extract_json(content)
