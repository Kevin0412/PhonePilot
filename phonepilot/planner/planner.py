"""Planner：自然语言 -> 结构化 Plan；运行期文本决策与视觉兜底。

- ``make_plan``       一句话 -> Plan（容错解析，无步骤则抛 PlanError）。
- ``decide``          运行期单步决策，mode="fast"（仅文字、关思考）或 "rich"（带坐标+图标、可思考）。
- ``recover_vision``  OCR 处理不了时看截图给归一化坐标。
所有运行期调用对异常一律降级为 wait，绝不让 LLM 故障中断主循环。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..types import ACTION_KINDS, Action, Observation, Plan, RiskLevel, Step
from . import context, prompts
from .llm_base import LLMClient, VisionLLMClient

log = logging.getLogger(__name__)


class PlanError(RuntimeError):
    """规划阶段失败（无法产出有效计划）。"""


_ACTION_ALIASES = {
    "click": "tap",
    "press": "tap",
    "longpress": "long_press",
    "long-press": "long_press",
    "input": "text",
    "type": "text",
    "none": "wait",
    "noop": "wait",
    "idle": "wait",
}
_DIRECTION_SUFFIX = ("up", "down", "left", "right")


def _coerce_point(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def action_from_dict(data: Any, *, default_reason: str = "") -> Action:
    """把 LLM 返回的动作 dict 容错解析为 Action（不做几何/安全校验，那在 loop/safety）。"""
    if not isinstance(data, dict):
        return Action(kind="wait", reason="LLM 返回非对象")

    raw = str(data.get("action", "")).strip().lower().replace(" ", "_")
    if not raw:
        log.warning("LLM 动作缺少 action 字段，降级为 wait：%s", data)
    direction = str(data.get("direction", "")).strip().lower()

    # 从 "scroll_down" / "swipe_up" 这类动作名推断方向
    for suffix in _DIRECTION_SUFFIX:
        if raw.endswith("_" + suffix):
            if not direction:
                direction = suffix
            raw = raw[: -(len(suffix) + 1)]
            break

    kind = _ACTION_ALIASES.get(raw, raw)
    if kind not in ACTION_KINDS:
        kind = "wait"

    index_raw = data.get("index")
    try:
        index = int(index_raw) if index_raw is not None else None
    except (TypeError, ValueError):
        index = None

    if kind == "text":
        text = str(data.get("input") or data.get("text") or "")
    elif kind == "launch":
        text = str(data.get("package") or data.get("text") or data.get("input") or "")
    elif kind == "intent":
        text = str(data.get("intent") or data.get("text") or data.get("input") or "")
    else:
        text = str(data.get("text") or "")

    action = Action(
        kind=kind,
        index=index,
        text=text,
        point=_coerce_point(data.get("point")),
        from_norm=_coerce_point(data.get("from") or data.get("from_norm")),
        to_norm=_coerce_point(data.get("to") or data.get("to_norm")),
        direction=direction if direction in _DIRECTION_SUFFIX else "",
        key=str(data.get("key") or ""),
        risk=RiskLevel.from_str(data.get("risk")),
        reason=str(data.get("reason") or default_reason),
    )
    # tap/long_press 必须有 index 或 point，否则会在执行层因「缺少坐标」抛错——降级为 wait
    if action.kind in ("tap", "long_press") and action.index is None and action.point is None:
        log.warning("%s 动作缺少 index 与 point，降级为 wait：%s", action.kind, data)
        return Action(kind="wait", reason="tap 缺少目标(index/point)")
    if action.kind in ("launch", "intent") and not action.text:
        log.warning("%s 动作缺少 包名/intent，降级为 wait：%s", action.kind, data)
        return Action(kind="wait", reason="launch/intent 缺少目标")
    return action


class Planner:
    def __init__(
        self,
        llm: LLMClient,
        vision: Optional[VisionLLMClient] = None,
        *,
        fast: Optional[dict] = None,
        rich: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.vision = vision
        self.fast = {"thinking": False, "max_tokens": 160, "temperature": 0.0, **(fast or {})}
        self.rich = {"thinking": True, "max_tokens": 512, "temperature": 0.0, **(rich or {})}

    # ---- 起始规划 ---- #
    def make_plan(self, task: str, *, screen_hint: str = "") -> Plan:
        user = f"用户任务：{task}"
        if screen_hint:
            user += f"\n\n当前屏幕文字（供参考）：\n{screen_hint}"
        last_err = "未知错误"
        for attempt in range(3):  # 模型偶发空响应/解析失败，重试
            try:
                data = self.llm.complete_json(
                    prompts.PLAN_SYSTEM_PROMPT,
                    user,
                    thinking=self.rich["thinking"],
                    # 规划是一次性调用，给足 token：推理模型(如 MiMo)的 reasoning_content 也吃额度
                    max_tokens=max(2048, int(self.rich["max_tokens"])),
                    temperature=0.0,
                )
                plan = Plan.from_dict(task, data)
                if plan.steps:
                    log.info("规划完成：%d 步，整体风险=%s", len(plan.steps), plan.overall_risk.value)
                    return plan
                last_err = f"无有效步骤：{data}"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
            log.warning("规划第 %d 次未成功（%s），重试", attempt + 1, last_err)
        raise PlanError(f"规划失败：{last_err}")

    # ---- 动态重规划：卡住/遇到未预料情况时，基于当前屏幕重新规划剩余步骤 ---- #
    def replan(self, task: str, obs: Observation, *, reason: str = "") -> Plan:
        from .context import format_fast

        hint = (
            f"（这是执行中途的重新规划）当前前台：{obs.foreground or '未知'}\n"
            f"当前屏幕元素（编号: 文字 @归一化位置）：\n{format_fast(obs.elements, obs.image_w, obs.image_h)}\n"
            f"遇到的情况：{reason}\n"
            "请基于【当前屏幕实况】，规划从现在起完成任务的剩余步骤"
            "（例如目标联系人不在会话列表，就先点搜索、输入名字、再进入聊天）。"
        )
        return self.make_plan(task, screen_hint=hint)

    # ---- 运行期文本决策 ---- #
    def decide(self, obs: Observation, plan: Plan, step: Step, step_idx: int, *, mode: str = "fast") -> Action:
        opts = self.fast if mode == "fast" else self.rich
        builder = context.build_user_fast if mode == "fast" else context.build_user_rich
        user = builder(plan, step, step_idx, obs)
        try:
            data = self.llm.complete_json(
                prompts.DECIDE_SYSTEM_PROMPT,
                user,
                thinking=bool(opts["thinking"]),
                max_tokens=int(opts["max_tokens"]),
                temperature=float(opts["temperature"]),
            )
        except Exception as exc:  # noqa: BLE001 - 降级为 wait，不中断主循环
            log.warning("LLM decide(%s) 失败：%s", mode, exc)
            return Action(kind="wait", reason=f"llm-{mode} error: {exc}")
        return action_from_dict(data, default_reason=f"{mode}-llm")

    # ---- 逐步 LLM 驱动：每个动作都看当前屏幕决策（ReAct GUI agent） ---- #
    def act(self, task: str, obs: Observation, history: list[str], *, mode: str = "fast", image=None) -> Action:
        # 默认用紧凑的「文字+坐标」上下文（比逐元素 JSON 小很多，更不易触发空/超时响应）
        screen = (
            context.format_rich(obs.elements, obs.image_w, obs.image_h)
            if mode == "rich"
            else context.format_fast(obs.elements, obs.image_w, obs.image_h)
        )
        hist = "\n".join(history[-8:]) if history else "（无）"
        user = (
            f"总任务：{task}\n当前前台：{obs.foreground or '未知'}\n"
            f"已执行动作（最近在后）：\n{hist}\n\n"
            f"屏幕尺寸：{obs.image_w}x{obs.image_h}；元素后 @(x,y) 为归一化位置。\n"
            f"当前屏幕元素：\n{screen}\n\n请决定下一个动作。"
        )
        opts = self.rich if mode == "rich" else self.fast
        last_err = "?"
        for _ in range(3):  # 模型偶发空/截断响应，重试
            try:
                data = self.llm.complete_json(
                    prompts.ACT_SYSTEM_PROMPT, user,
                    thinking=bool(opts["thinking"]), max_tokens=max(800, int(opts["max_tokens"])), temperature=0.0,
                    image=image,
                )
                return action_from_dict(data, default_reason="llm-act")
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
        log.warning("LLM act 重试后仍失败：%s", last_err)
        return Action(kind="wait", reason=f"act error: {last_err}")

    # ---- 视觉定位控件（图像识别框控件）：返回归一化坐标 ---- #
    def locate(self, image: Any, what: str) -> Optional[tuple[float, float]]:
        if self.vision is None:
            return None
        try:
            data = self.vision.decide_json(
                prompts.LOCATE_SYSTEM_PROMPT,
                f"定位控件：{what}。返回其中心归一化坐标 {{\"point\":[x,y]}}",
                image, max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("视觉定位失败：%s", exc)
            return None
        p = data.get("point")
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                return (float(p[0]), float(p[1]))
            except (TypeError, ValueError):
                return None
        return None

    # ---- 视觉兜底 ---- #
    def recover_vision(self, obs: Observation, image: Any, plan: Plan, step: Step, step_idx: int) -> Action:
        if self.vision is None:
            return Action(kind="wait", reason="未配置视觉客户端")
        user = context.build_user_vision(plan, step, step_idx)
        try:
            data = self.vision.decide_json(
                prompts.VISION_SYSTEM_PROMPT, user, image, max_tokens=int(self.rich["max_tokens"])
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("视觉决策失败：%s", exc)
            return Action(kind="wait", reason=f"vision error: {exc}")
        return action_from_dict(data, default_reason="vision")
