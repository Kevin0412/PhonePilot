"""视觉多模态 LLM 适配器（OCR 兜底）。

DeepSeek 是纯文本，OCR 处理不了的页面（图标/图片/复杂布局）升级到视觉模型「看截图给坐标」。
用一个 OpenAI 兼容客户端经 provider 预置切换 Qwen-VL / Kimi / GLM / MiMo —— 截图以 base64
data URL 走 OpenAI 的 image_url 内容格式。各家端点/模型名以实测为准（GLM v4 可能需微调）。
cv2 仅在编码截图时懒加载。
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional

from .llm_base import VisionLLMClient, extract_json, http_post_json

# provider -> (base_url, default_model, env_key)。端点实现/联调时请核对。
VISION_PRESETS: dict[str, tuple[Optional[str], str, str]] = {
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max", "DASHSCOPE_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "glm-4v-plus", "GLM_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "moonshot-v1-8k-vision-preview", "MOONSHOT_API_KEY"),
    "mimo": ("https://token-plan-cn.xiaomimimo.com/v1", "mimo-v2.5", "MIMO_API_KEY"),
}


class OpenAICompatVisionClient(VisionLLMClient):
    def __init__(
        self,
        *,
        provider: str = "qwen",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        preset_url, preset_model, env_key = VISION_PRESETS.get(provider, (None, "", "MIMO_API_KEY"))
        self.base_url = base_url or preset_url
        self.model = model or preset_model
        self.api_key = api_key or os.environ.get(env_key)
        self.timeout = timeout
        if not self.base_url:
            raise RuntimeError(f"视觉 provider={provider} 未配置 base_url（在 config.vision.base_url 填写）")
        if not self.model:
            raise RuntimeError(f"视觉 provider={provider} 未配置 model")
        if not self.api_key:
            raise RuntimeError(f"缺少视觉 API key（环境变量 {env_key}）")

    @staticmethod
    def _encode_png(image: Any) -> str:
        import cv2  # 懒加载

        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("截图 PNG 编码失败")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def decide_json(
        self,
        system: str,
        user: str,
        image: Any,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        b64 = self._encode_png(image)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = http_post_json(self.base_url, "/chat/completions", headers, payload, self.timeout)
        content = resp["choices"][0]["message"]["content"]
        if isinstance(content, list):  # 个别实现返回内容块列表
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return extract_json(content)


class MockVisionClient(VisionLLMClient):
    """测试/离线用：返回预置动作（默认 wait）。"""

    def __init__(self, *, responses: Optional[list[dict]] = None, default: Optional[dict] = None) -> None:
        self.responses = list(responses or [])
        self.default = default or {"action": "wait", "reason": "mock vision default"}
        self.calls = 0

    def decide_json(self, system: str, user: str, image: Any, *, max_tokens: int = 512, temperature: float = 0.0) -> dict:
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return dict(self.default)
