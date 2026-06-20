"""本地规则状态机（毫秒级、零网络）——高频决策的主力。

按优先级：错误恢复 > 加载等待 > 当前子目标的候选控件 > 滚动翻找 > 等待。
与参考项目最大的差异：接收 ``step`` 参数，按**当前子目标**的 ``target_markers`` 动态选目标，
而非隐式单目标。返回的 Action 仅供决策，几何/安全校验在 loop/safety 完成。
"""

from __future__ import annotations

from ..perception.signature import contains, find_candidate
from ..profile import Profile
from ..types import Action, ScreenElement, Step


def local_rule_decision(
    elements: list[ScreenElement],
    step: Step,
    profile: Profile,
    *,
    allow_scroll: bool = True,
) -> Action:
    # 1. 错误恢复：命中错误上下文时优先点击「重试/确定」等
    rec_ctx = contains(elements, profile.recovery_markers)
    if rec_ctx:
        hit = find_candidate(elements, profile.recovery_actions, reject_markers=profile.forbidden_markers)
        if hit:
            idx, el, marker = hit
            return Action(
                kind="tap", index=idx, text=el.text,
                reason=f"local recovery: {marker}（上下文={rec_ctx}）",
            )

    # 2. 加载/排队 -> 等待
    loading = contains(elements, profile.loading_markers)
    if loading:
        return Action(kind="wait", reason=f"loading: {loading}")

    # 3. 当前子目标的候选控件
    if step.target_markers:
        hit = find_candidate(elements, step.target_markers, reject_markers=profile.forbidden_markers)
        if hit:
            idx, el, marker = hit
            return Action(
                kind="tap", index=idx, text=el.text, risk=step.risk,
                reason=f"local target: {marker}",
            )
        # 4. scroll-to-find：候选不在当前屏幕，向下翻找
        if allow_scroll:
            return Action(kind="scroll", direction="down", reason="scroll-to-find: 候选不在当前屏幕")

    # 5. 无明确本地动作（交由上层决定是否升级 LLM）
    return Action(kind="wait", reason="no local action found")
