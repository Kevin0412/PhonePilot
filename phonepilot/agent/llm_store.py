"""单槽并发 LLM 存储（移植参考项目 RuntimeLlmStore）。

关键工程决策：同一时刻只跑一个请求，忙时新请求直接丢弃（不排队）；对同一页面签名按
``min_interval`` 节流；结果带「新鲜度」，过期即视为无效。这样 LLM 永远不会成为主循环
的延迟瓶颈——它在后台异步算，主循环在随后的若干帧里把结果捡起来用。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ..types import Action

log = logging.getLogger(__name__)


class RuntimeLlmStore:
    def __init__(self, name: str = "llm", *, min_interval: float = 3.0) -> None:
        self.name = name
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._active: Optional[str] = None
        self._last_started: dict[str, float] = {}
        self._results: dict[str, tuple[float, Action]] = {}

    def start(self, key: str, worker: Callable[[], Action]) -> bool:
        """尝试启动后台 worker。忙/节流则返回 False（请求被丢弃）。"""
        now = time.monotonic()
        with self._lock:
            if self._active is not None:
                return False
            if now - self._last_started.get(key, 0.0) < self.min_interval:
                return False
            self._active = key
            self._last_started[key] = now

        def run() -> None:
            try:
                result = worker()
                with self._lock:
                    self._results[key] = (time.monotonic(), result)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s worker 失败：%s", self.name, exc)
            finally:
                with self._lock:
                    self._active = None

        threading.Thread(target=run, name=f"pp-{self.name}", daemon=True).start()
        return True

    def get(self, key: str, max_age: float) -> Optional[Action]:
        """取结果（超 max_age 视为过期返回 None）。"""
        with self._lock:
            rec = self._results.get(key)
        if rec is None:
            return None
        ts, action = rec
        if time.monotonic() - ts > max_age:
            return None
        return action
