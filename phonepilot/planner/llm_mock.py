"""Mock 文本 LLM（测试/离线/演示）。

依据 system 提示词中的独有短语路由：含「任务拆解」=规划调用，返回预置 Plan JSON；
否则=单步决策调用，依次弹出预置决策，用尽后返回默认（wait）。
"""

from __future__ import annotations

from typing import Optional

from .llm_base import LLMClient


class MockLLMClient(LLMClient):
    def __init__(
        self,
        *,
        plan: Optional[dict] = None,
        decisions: Optional[list[dict]] = None,
        default_decision: Optional[dict] = None,
    ) -> None:
        self.plan = plan
        self.decisions = list(decisions or [])
        self.default_decision = default_decision or {"action": "wait", "reason": "mock default"}
        self.calls: list[tuple[str, str]] = []  # (kind, user) 供测试断言

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.0,
        image=None,
    ) -> dict:
        is_plan = "任务拆解" in system
        self.calls.append(("plan" if is_plan else "decide", user))
        if is_plan:
            if self.plan is None:
                raise RuntimeError("MockLLMClient 未提供 plan")
            return dict(self.plan)
        if self.decisions:
            return self.decisions.pop(0)
        return dict(self.default_decision)
