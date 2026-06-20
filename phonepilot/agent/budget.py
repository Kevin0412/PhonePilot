"""执行预算与超时阈值。

参考项目用细粒度的步数/时长/重试上限防止失控。这里集中管理：动作总数、恢复/结构化/
视觉调用次数、滚动次数、无变化重试、未知页/加载超时、总时长。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_steps: int = 30
    max_recover_calls: int = 6
    max_rich_calls: int = 8
    max_vision_calls: int = 4
    max_no_change_retries: int = 2
    max_scrolls: int = 6
    unknown_stop_seconds: float = 8.0
    loading_stop_seconds: float = 45.0
    total_timeout_s: float = 600.0

    # 运行计数（执行期递增）
    actions: int = 0
    rich_calls: int = 0
    vision_calls: int = 0
    scrolls: int = 0
    _start: float = field(default_factory=time.monotonic)

    def tick(self) -> None:
        self.actions += 1

    def exceeded(self) -> str | None:
        """超预算则返回原因字符串，否则 None。"""
        if self.actions >= self.max_steps:
            return f"达到最大动作数 {self.max_steps}"
        if time.monotonic() - self._start >= self.total_timeout_s:
            return f"超过总时长 {self.total_timeout_s}s"
        return None

    def rich_ok(self) -> bool:
        return self.rich_calls < self.max_rich_calls

    def vision_ok(self) -> bool:
        return self.vision_calls < self.max_vision_calls

    def scroll_ok(self) -> bool:
        return self.scrolls < self.max_scrolls
