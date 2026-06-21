"""LLM 抽象接口（文本 + 视觉）与 JSON 容错抽取。

所有客户端用标准库 ``http.client`` 直连，不引入任何 SDK —— 与参考项目一致，
也便于原生 App 阶段换成 OkHttp 时契约不变。``complete_json`` / ``decide_json``
对外只暴露「已解析的 dict」，模型偶发的解释性文字由 ``extract_json`` 兜底剥离。
"""

from __future__ import annotations

import abc
import http.client
import json
import re
from typing import Any
from urllib.parse import urlparse

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def http_post_json(
    base_url: str,
    path: str,
    headers: dict,
    payload: dict,
    timeout: float,
    *,
    retries: int = 2,
) -> dict:
    """向 ``base_url`` + ``path`` POST 一个 JSON，返回响应 JSON。失败重试。

    标准库 http.client 实现，无第三方依赖。``base_url`` 的路径前缀（如 ``/v1`` 或
    ``/compatible-mode/v1``）会与 ``path`` 拼接。
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    u = urlparse(base_url)
    use_https = u.scheme != "http"
    prefix = u.path.rstrip("/") if u.path and u.path != "/" else ""
    full_path = prefix + path
    hdrs = {"Content-Type": "application/json", **headers}
    for hk, hv in hdrs.items():  # 提前给出清晰错误，而不是 http.client 的 latin-1 崩溃
        if not str(hv).isascii():
            raise RuntimeError(f"请求头 {hk} 含非 ASCII 字符（多半是 .env 里的 API key 没填对）")
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        conn = None
        try:
            ctor = http.client.HTTPSConnection if use_https else http.client.HTTPConnection
            conn = ctor(u.hostname, u.port, timeout=timeout)
            conn.request("POST", full_path, body=body, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:300]}")
            return json.loads(raw)
        except (http.client.HTTPException, OSError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        finally:
            if conn is not None:
                conn.close()
    raise RuntimeError(f"LLM 请求失败（{base_url}{path}）：{last_error}")


def extract_json(raw: str) -> dict:
    """从模型输出中稳健地抽取首个 JSON 对象。

    依次尝试：直接解析 → 去除 ```json 围栏后解析 → 花括号配对扫描出首个完整对象。
    全部失败抛 ValueError。
    """
    if raw is None:
        raise ValueError("空响应")
    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 花括号配对扫描，定位首个完整 {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"无法从响应中解析 JSON: {raw[:200]!r}")


class LLMClient(abc.ABC):
    """文本 LLM：输入 system/user，返回解析后的 JSON dict。"""

    @abc.abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        raise NotImplementedError


class VisionLLMClient(abc.ABC):
    """视觉多模态 LLM：输入 system/user + 截图，返回解析后的 JSON dict。"""

    @abc.abstractmethod
    def decide_json(
        self,
        system: str,
        user: str,
        image: Any,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        raise NotImplementedError
