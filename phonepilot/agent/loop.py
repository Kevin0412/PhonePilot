"""Agent 主循环：感知 → 本地决策 → (卡住升级 LLM) → 安全校验 → RiskGate → 执行 → 等待变化。

返回码：0 = 任务完成；2 = 停下交人工 / 未完成（命中硬停止、预算耗尽、无响应、被拒绝等）。
LLM 只在本地规则「无明确动作且卡住」时按 fast→rich→vision 阶梯异步升级，绝不参与每帧。
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..perception.signature import contains
from ..profile import Profile
from ..types import Action, Decision, Observation, Plan
from .budget import Budget
from .gestures import execute_action
from .llm_store import RuntimeLlmStore
from .local_rules import local_rule_decision
from .observe import observe
from .safety import SafetyError, validate_action, validate_point

log = logging.getLogger(__name__)

DONE = 0
HANDOVER = 2


@dataclass
class LoopConfig:
    poll_interval: float = 0.3
    after_click_delay: float = 0.4
    transition_timeout: float = 12.0
    transition_threshold: float = 0.025
    llm_max_age: float = 4.0
    vision_max_age: float = 6.0
    llm_min_interval: float = 3.0
    foreground_interval: float = 2.0


# --------------------------------------------------------------------------- #
# 屏幕变化检测（移植参考项目）
# --------------------------------------------------------------------------- #
def screen_distance(a: Any, b: Any) -> float:
    """两帧的归一化灰度差均值（0~1）。形状不一致视为完全不同。"""
    import numpy as np

    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255.0


def wait_for_change(executor, baseline: Any, *, timeout: float, threshold: float, poll: float = 0.2) -> Optional[Any]:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        cur = executor.screencap()
        if screen_distance(baseline, cur) >= threshold:
            return cur
        time.sleep(poll)
    return None


# --------------------------------------------------------------------------- #
# 审计落盘（best-effort，cv2 缺失时仅存 JSON）
# --------------------------------------------------------------------------- #
def _save_image(run_dir: Optional[Path], name: str, image: Any) -> None:
    if run_dir is None or image is None:
        return
    try:
        import cv2

        run_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(run_dir / f"{name}.png"), image)
    except Exception as exc:  # noqa: BLE001
        log.debug("保存截图失败：%s", exc)


def _save_audit(run_dir: Optional[Path], name: str, image: Any, obs: Observation, note: dict) -> None:
    if run_dir is None:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": note,
            "signature": obs.signature,
            "foreground": obs.foreground,
            "elements": [e.to_dict() for e in obs.elements],
        }
        (run_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.debug("保存审计失败：%s", exc)
    _save_image(run_dir, name, image)


def _stamp() -> str:
    return _dt.datetime.now().strftime("%H%M%S_%f")[:-3]


def manual_stop(executor, obs: Optional[Observation], reason: str, *, run_dir: Optional[Path], image: Any = None) -> int:
    log.warning("⏸ 停下交人工：%s", reason)
    try:
        img = image if image is not None else executor.screencap()
        _save_image(run_dir, f"{_stamp()}_STOP", img)
    except Exception:  # noqa: BLE001
        pass
    try:
        print("\a", end="", flush=True)  # 蜂鸣提醒
    except Exception:  # noqa: BLE001
        pass
    return HANDOVER


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
def run_agent(
    plan: Plan,
    *,
    executor,
    ocr,
    planner,
    gate,
    profile: Profile,
    budget: Budget,
    loop: LoopConfig,
    icons=None,
    vision_enabled: bool = False,
    run_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    fast_store = RuntimeLlmStore("fast", min_interval=loop.llm_min_interval)
    rich_store = RuntimeLlmStore("rich", min_interval=loop.llm_min_interval)
    vision_store = RuntimeLlmStore("vision", min_interval=loop.llm_min_interval)

    step_idx = 0
    last_sig = ""
    stuck_frames = 0
    no_change = 0
    unknown_since: Optional[float] = None
    loading_since: Optional[float] = None
    final_wait_since: Optional[float] = None
    step_entered_sig: Optional[str] = None   # 进入当前步时的页面签名（用于「确实导航过」判定）
    launched_step_idx = -1                    # 已用命令启动过 app 的步序号（避免重复启动）

    def now() -> float:
        return time.monotonic()

    def escalate(obs: Observation, image: Any, ocr_blind: bool):
        """卡住时按 fast→rich→vision 升级，返回 (Action, source) 或 None。"""
        sig = obs.signature
        best: Optional[tuple[Action, str]] = None

        fast_store.start(f"{sig}|fast", lambda: planner.decide(obs, plan, plan.steps[step_idx], step_idx, mode="fast"))
        a = fast_store.get(f"{sig}|fast", loop.llm_max_age)
        if a is not None and a.kind != "wait":
            best = (a, "text-llm-fast")

        if stuck_frames >= 3 and budget.rich_ok():
            if rich_store.start(f"{sig}|rich", lambda: planner.decide(obs, plan, plan.steps[step_idx], step_idx, mode="rich")):
                budget.rich_calls += 1
            a = rich_store.get(f"{sig}|rich", loop.llm_max_age)
            if a is not None and a.kind != "wait":
                best = (a, "text-llm-rich")

        if vision_enabled and (ocr_blind or stuck_frames >= 5) and budget.vision_ok():
            if vision_store.start(f"{sig}|vision", lambda: planner.recover_vision(obs, image, plan, plan.steps[step_idx], step_idx)):
                budget.vision_calls += 1
            a = vision_store.get(f"{sig}|vision", loop.vision_max_age)
            if a is not None and a.kind != "wait":
                best = (a, "vision")

        return best

    while True:
        exc = budget.exceeded()
        if exc:
            return manual_stop(executor, None, f"预算耗尽：{exc}", run_dir=run_dir)

        image, obs = observe(executor, ocr, icons, with_foreground=True)

        # 1) 硬停止：支付/验证码/成功等页面 -> 交人工
        m = contains(obs.elements, profile.hard_stop_markers)
        if m:
            _save_audit(run_dir, f"{_stamp()}_hardstop", image, obs, {"reason": m})
            return manual_stop(executor, obs, f"命中硬停止标志：{m}", run_dir=run_dir, image=image)

        # 2) 完成判据：连续推进已满足的子目标。
        #    防误判：只有「进入本步后页面确实变化过(即真导航了)」或本步无可点目标时，才允许用
        #    success_markers 判完成——否则上级菜单里的入口名(如「WLAN」)会被当成「已进入 WLAN 页」。
        while step_idx < len(plan.steps):
            cur = plan.steps[step_idx]
            if step_entered_sig is None:
                step_entered_sig = obs.signature
            navigated = obs.signature != step_entered_sig
            # 命令驱动的步骤用「导航/前台」判完成，比文字标志可靠（文字跨 ROM 易变：可用网络 vs 可用WLAN）
            pkg_open = bool(cur.package) and cur.package in (obs.foreground or "")
            intent_reached = bool(cur.intent) and launched_step_idx == step_idx and navigated
            done_here = pkg_open or intent_reached or (
                bool(cur.success_markers)
                and contains(obs.elements, cur.success_markers)
                and (navigated or not cur.target_markers)
            )
            if done_here:
                log.info("✓ 子目标完成（%d/%d）：%s", step_idx + 1, len(plan.steps), cur.goal)
                step_idx += 1
                step_entered_sig = None
            else:
                break

        if step_idx >= len(plan.steps):
            # 末步是命令驱动(intent/package)的，达成即视作完成——最终文字标志跨 ROM 易变，不强求
            last_cmd = bool(plan.steps) and bool(plan.steps[-1].intent or plan.steps[-1].package)
            if not plan.final_success_markers or last_cmd or contains(obs.elements, plan.final_success_markers):
                log.info("✅ 任务完成：%s", plan.task)
                _save_audit(run_dir, f"{_stamp()}_DONE", image, obs, {"task": plan.task})
                return DONE
            final_wait_since = final_wait_since or now()
            if now() - final_wait_since >= budget.unknown_stop_seconds:
                return manual_stop(executor, obs, "所有子目标完成但最终标志未出现", run_dir=run_dir, image=image)
            last_sig = obs.signature
            time.sleep(loop.poll_interval)
            continue

        step = plan.steps[step_idx]

        # 2.5) 能用命令就别点：本步若给了 intent/包名，直接命令直达/启动，不点图标（更稳）
        if launched_step_idx != step_idx and (step.intent or step.package):
            if dry_run:
                log.info("[dry-run] 下一步：命令打开 %s（不点击）", step.intent or step.package)
                return DONE
            launched_step_idx = step_idx
            try:
                if step.intent:
                    executor.start_intent(step.intent)
                    log.info("▶ 命令直达(intent)：%s", step.intent)
                else:
                    executor.launch_app(step.package)
                    log.info("▶ 命令启动应用：%s", step.package)
                budget.tick()
                last_sig = ""
                time.sleep(loop.after_click_delay)
                continue
            except Exception as e:  # noqa: BLE001 - 命令失败回退到点击
                log.warning("命令打开失败（%s），回退点击：%s", step.intent or step.package, e)

        # 3) 本地规则决策
        action = local_rule_decision(obs.elements, step, profile, allow_scroll=budget.scroll_ok())
        source = "local"
        is_loading = action.reason.startswith("loading")

        # 卡住计数（同一签名、本地无动作、非加载）
        if action.kind == "wait" and not is_loading:
            stuck_frames = stuck_frames + 1 if obs.signature == last_sig else 1
            ocr_blind = not any((e.text or "").strip() for e in obs.elements)
            esc = escalate(obs, image, ocr_blind)
            if esc is not None:
                action, source = esc
        else:
            stuck_frames = 0

        # dry-run：只预览当前屏幕的下一步决策，不执行也不推进
        if dry_run:
            log.info(
                "[dry-run] 下一步[%s]：kind=%s index=%s text=%r dir=%s | 原因=%s",
                source, action.kind, action.index, action.text, action.direction or "-", action.reason,
            )
            _save_audit(
                run_dir, f"{_stamp()}_dryrun", image, obs,
                {"source": source, "kind": action.kind, "text": action.text, "reason": action.reason},
            )
            return DONE

        # 4) 终态/等待分支
        if action.kind == "stop":
            return manual_stop(executor, obs, f"[{source}] 决策停止：{action.reason}", run_dir=run_dir, image=image)
        if action.kind == "done":
            log.info("✅ [%s] 决策判定任务完成", source)
            return DONE
        if action.kind == "wait":
            if is_loading:
                loading_since = loading_since or now()
                if now() - loading_since >= budget.loading_stop_seconds:
                    return manual_stop(executor, obs, "加载状态超时", run_dir=run_dir, image=image)
            else:
                loading_since = None
                if obs.signature == last_sig:
                    unknown_since = unknown_since or now()
                    if now() - unknown_since >= budget.unknown_stop_seconds:
                        return manual_stop(executor, obs, "未知页面等待超时", run_dir=run_dir, image=image)
                else:
                    unknown_since = None
            last_sig = obs.signature
            time.sleep(loop.poll_interval)
            continue

        # 5) 解析执行参数 + 安全校验 + RiskGate
        #    注意：等待计时器(unknown_since/loading_since)在「真正执行后」才清空，
        #    否则安全校验失败 continue 会把卡顿计时不一致地重置（见下方执行成功处）。
        point: Optional[tuple[int, int]] = None
        target_text = ""
        decision = Decision(action, source)

        if action.uses_index:
            try:
                target = validate_action(action, obs.elements, profile.forbidden_markers)
            except SafetyError as e:
                log.warning("丢弃不安全动作：%s", e)
                last_sig = obs.signature
                time.sleep(loop.poll_interval)
                continue
            if gate.is_forbidden(target.text):
                return manual_stop(executor, obs, f"目标在黑名单：{target.text}", run_dir=run_dir, image=image)
            point, target_text = target.center, target.text
            if not gate.approve(action, target_text, is_coordinate=False):
                return manual_stop(executor, obs, f"动作被拒绝：{target_text}", run_dir=run_dir, image=image)

        elif action.kind in ("tap", "long_press") and action.point:
            try:
                point = validate_point(action.point, (obs.image_w, obs.image_h))
            except SafetyError as e:
                log.warning("坐标无效：%s", e)
                last_sig = obs.signature
                time.sleep(loop.poll_interval)
                continue
            target_text = f"坐标{point}"
            if not gate.approve(action, target_text, is_coordinate=True):
                return manual_stop(executor, obs, "坐标点击被拒绝", run_dir=run_dir, image=image)

        else:
            # 手势/文本/导航
            if action.kind == "text":
                if not gate.approve(action, action.text, is_coordinate=False):
                    return manual_stop(executor, obs, "文本输入被拒绝", run_dir=run_dir, image=image)
            if action.kind in ("scroll", "fling", "swipe", "drag"):
                if not budget.scroll_ok():
                    return manual_stop(executor, obs, "滚动次数超预算（可能找不到目标）", run_dir=run_dir, image=image)
                budget.scrolls += 1

        # 6) 审计 + 执行
        _save_audit(
            run_dir, f"{_stamp()}_act", image, obs,
            {"source": source, "kind": action.kind, "target": target_text, "reason": action.reason},
        )
        try:
            desc = execute_action(executor, action, point=point, image_wh=(obs.image_w, obs.image_h))
        except SafetyError as e:
            log.warning("动作被执行层拒绝（缺少坐标/不可执行）：%s", e)
            last_sig = obs.signature
            time.sleep(loop.poll_interval)
            continue
        unknown_since = loading_since = None  # 已实际行动，清空等待计时
        log.info("▶ 执行[%s]：%s | 原因=%s", source, desc, action.reason)
        budget.tick()

        # 7) 导航类手势不强制「页面变化」断言
        if action.is_nav_gesture:
            last_sig = ""
            stuck_frames = 0
            time.sleep(loop.after_click_delay)
            continue

        # 8) 等待页面变化 / 无响应重试
        changed = wait_for_change(executor, image, timeout=loop.transition_timeout, threshold=loop.transition_threshold)
        if changed is None:
            no_change += 1
            if no_change > budget.max_no_change_retries:
                return manual_stop(executor, obs, f"点击后无变化：{target_text or action.kind}", run_dir=run_dir, image=image)
            log.warning("点击后无变化，重试 %d/%d", no_change, budget.max_no_change_retries)
            last_sig = ""
            unknown_since = loading_since = None
            time.sleep(loop.after_click_delay)
            continue

        no_change = 0
        last_sig = ""
        stuck_frames = 0
        time.sleep(loop.after_click_delay)
