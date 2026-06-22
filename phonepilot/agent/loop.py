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

from ..perception.signature import contains, find_candidate
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


def _find_input_element(obs: Observation) -> Optional[tuple[int, int]]:
    """从 observe 已融合的元素里找底部输入框：kind=ui/edit + 底部(>0.80h) + 宽(>0.30w)。"""
    h, w = obs.image_h, obs.image_w
    best = None
    for e in obs.elements:
        if e.kind not in ("ui", "edit", "control"):
            continue
        bx1, by1, bx2, by2 = e.bounds
        cy = (by1 + by2) / 2
        bw = bx2 - bx1
        if cy > 0.80 * h and bw > 0.30 * w and (by2 - by1) < 0.12 * h:
            if best is None or cy > best[1]:
                best = e.center
    return best


def _focus_input(executor, obs, target_markers, profile, *, planner=None, image=None, point=None) -> tuple[int, int]:
    """聚焦输入框，分层检测（最准+最便宜优先）：
    1) UIAutomator EditText（标准 app 最精确；QQ/微信自绘拿不到）
    2) OCR 占位符文字（target_markers）
    3) 检测器/控件树产出的底部宽矮框（UI 检测器能可靠框出低对比输入框）
    4) LLM 给的坐标 point
    5) CV 图标行反推底栏（input_bar_center）
    6) 位置兜底 0.40w×0.955h（只在全部失败时用）
    """
    from ..perception import cv_controls, uia

    xy: Optional[tuple[int, int]] = None
    try:
        xy = uia.input_center(executor.ui_dump())
    except Exception:  # noqa: BLE001
        xy = None
    if xy is None and target_markers:
        hit = find_candidate(obs.elements, target_markers, reject_markers=profile.forbidden_markers)
        if hit:
            xy = hit[1].center
    if xy is None:
        xy = _find_input_element(obs)
    if xy is None and point:
        try:
            xy = validate_point(point, (obs.image_w, obs.image_h))
        except SafetyError:
            xy = None
    if xy is None and image is not None:
        xy = cv_controls.input_bar_center(image)
    if xy is None:
        xy = (int(obs.image_w * 0.40), int(obs.image_h * 0.955))
    executor.tap(*xy)
    return xy


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
    ui_detect=None,
    vision_enabled: bool = False,
    with_cv: bool = False,
    with_ui: bool = False,
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
    typed_step_idx = -1                       # 已打字的步序号（避免重复输入）
    replans = 0                               # 已重规划次数

    def now() -> float:
        return time.monotonic()

    def try_replan(observation: Observation, reason: str) -> bool:
        """卡住/遇未预料情况时，基于当前屏幕重新规划剩余步骤。成功返回 True（调用方 continue）。"""
        nonlocal plan, step_idx, step_entered_sig, launched_step_idx, typed_step_idx
        nonlocal last_sig, stuck_frames, unknown_since, loading_since, final_wait_since, replans
        if replans >= budget.max_replans:
            return False
        replans += 1
        log.info("↻ 第 %d 次重新规划（%s）", replans, reason)
        try:
            plan = planner.replan(plan.task, observation, reason=reason)
        except Exception as e:  # noqa: BLE001
            log.warning("重规划失败：%s", e)
            return False
        step_idx = 0
        step_entered_sig = None
        launched_step_idx = typed_step_idx = -1
        last_sig = ""
        stuck_frames = 0
        unknown_since = loading_since = final_wait_since = None
        return True

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

        image, obs = observe(executor, ocr, icons, ui_detect=ui_detect, with_foreground=True, with_cv=with_cv, with_ui=with_ui)

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
            typed_done = bool(cur.input_text) and typed_step_idx == step_idx
            done_here = pkg_open or intent_reached or typed_done or (
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
                if try_replan(obs, "步骤走完但任务似乎未达成"):
                    continue
                return manual_stop(executor, obs, "所有子目标完成但最终标志未出现", run_dir=run_dir, image=image)
            last_sig = obs.signature
            time.sleep(loop.poll_interval)
            continue

        step = plan.steps[step_idx]

        # 2.4) 前置校验：执行本步动作前，要求 verify 文字全部出现（如发送前确认在「王信」的聊天里）。
        #      操作不可靠，校验不过就别动手——优先重规划自我纠正，无预算则交人工，绝不在错误界面执行(尤其高风险)。
        if step.verify and not all(contains(obs.elements, [m]) for m in step.verify):
            missing = [m for m in step.verify if not contains(obs.elements, [m])]
            if try_replan(obs, f"前置校验失败：当前界面缺少 {missing}（步骤：{step.goal}）"):
                continue
            return manual_stop(
                executor, obs, f"前置校验失败，当前界面不符预期（需 {step.verify}），停止避免误操作",
                run_dir=run_dir, image=image,
            )

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

        # 2.6) 输入文字步骤：先聚焦输入框（若给了 target_markers），再用输入法打字（中文走 ADBKeyboard）
        if step.input_text and typed_step_idx != step_idx:
            if dry_run:
                log.info("[dry-run] 下一步：输入文字 %r（不执行）", step.input_text)
                return DONE
            if not gate.approve(Action(kind="text", text=step.input_text), step.input_text):
                return manual_stop(executor, obs, "文本输入被拒绝", run_dir=run_dir, image=image)
            executor.prepare_input()  # 先就绪输入法，再聚焦（顺序关键，否则切换输入法会丢焦点）
            _focus_input(executor, obs, step.target_markers, profile, planner=planner, image=image)
            time.sleep(0.6)
            try:
                executor.text(step.input_text)
                log.info("⌨ 输入文字：%r", step.input_text)
            except Exception as e:  # noqa: BLE001
                log.warning("输入失败：%s", e)
            typed_step_idx = step_idx
            budget.tick()
            last_sig = ""
            time.sleep(max(loop.after_click_delay, 1.2))  # 等搜索结果/界面刷新出来，避免下一步过早滚动
            continue

        # 3) 本地规则决策
        action = local_rule_decision(
            obs.elements, step, profile, allow_scroll=budget.scroll_ok(), image_h=obs.image_h
        )
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
                        if try_replan(obs, f"在「{step.goal}」卡住（未知页面）"):
                            continue
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
                if try_replan(obs, f"点击「{target_text or action.kind}」后无变化"):
                    no_change = 0
                    continue
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


def run_llm_agent(
    task: str,
    *,
    executor,
    ocr,
    planner,
    gate,
    profile: Profile,
    budget: Budget,
    loop: LoopConfig,
    icons=None,
    ui_detect=None,
    with_cv: bool = False,
    with_ui: bool = False,
    run_dir: Optional[Path] = None,
    dry_run: bool = False,
    mode: str = "fast",
) -> int:
    """逐步 LLM 驱动模式：每个动作都把当前屏幕(OCR+坐标)交给 LLM 决策，更灵活。

    安全/风险/硬停与 hybrid 模式一致：发送等高风险仍需确认，命中支付/验证码硬停。
    失败的 LLM 调用不计入动作预算（只计真正执行的动作），但连续失败到上限会停。
    """
    history: list[str] = []
    no_change = 0
    consec_fail = 0
    recovery_back = 0  # 连续「已在前台仍想重启 -> 改按返回回退」次数；做了正常动作就清零
    _last_action_key = ""  # 语义重复检测：kind+text
    _repeat_count = 0
    _action_window: list[str] = []  # 最近 N 步的 action_key，检测交替循环
    _last_caption_sig = ""  # 上次做 VL 标注的页面签名，避免同页重复调
    while True:
        exc = budget.exceeded()
        if exc:
            return manual_stop(executor, None, f"预算耗尽：{exc}", run_dir=run_dir)

        image, obs = observe(executor, ocr, icons, ui_detect=ui_detect, with_foreground=True, with_cv=with_cv, with_ui=with_ui,
                             vl_caption=(ui_detect is not None and obs.signature != _last_caption_sig) if False else False)
        # VL 标注放到 observe 外面：先拿 signature，签名变了才调（避免同页重复调 VL）
        if ui_detect is not None and obs.signature != _last_caption_sig:
            unlabeled = [e for e in obs.elements if e.text in ("[控件]", "") and e.kind in ("ui", "control")]
            if unlabeled:
                try:
                    from ..perception.vl_caption import caption_controls
                    obs = Observation(image_w=obs.image_w, image_h=obs.image_h,
                                     elements=caption_controls(image, obs.elements),
                                     signature=obs.signature, foreground=obs.foreground)
                except Exception as exc:  # noqa: BLE001
                    log.debug("VL 标注失败：%s", exc)
            _last_caption_sig = obs.signature
        m = contains(obs.elements, profile.hard_stop_markers)
        if m:
            return manual_stop(executor, obs, f"命中硬停止标志：{m}", run_dir=run_dir, image=image)

        action = planner.act(task, obs, history, mode=mode, image=image)
        log.info(
            "LLM决策: %s index=%s text=%r point=%s dir=%s | 原因=%s",
            action.kind, action.index, action.text, action.point, action.direction, action.reason,
        )

        if action.kind == "wait":  # 含 LLM 失败降级——不计预算，但连续失败到上限就停
            consec_fail += 1
            if consec_fail > 8:
                return manual_stop(executor, obs, "LLM 连续多次无有效决策", run_dir=run_dir, image=image)
            time.sleep(loop.poll_interval)
            continue
        consec_fail = 0

        # 语义重复检测（两层）：
        # 1) 连续相同动作：tap:微信 x5 → 停
        # 2) 交替循环：tap:头像→back→tap:头像→back（最近8步只有≤2种动作）→ 停
        action_key = f"{action.kind}:{action.text or ''}"
        if action_key == _last_action_key:
            _repeat_count += 1
        else:
            _last_action_key = action_key
            _repeat_count = 1
        _action_window.append(action_key)
        if len(_action_window) > 8:
            _action_window.pop(0)

        _is_loop = False
        if _repeat_count >= 3 and action.kind not in ("done", "stop", "wait"):
            hint = (f"⚠ 你已连续{_repeat_count}次执行相同动作({action.kind} {action.text!r})，"
                    f"说明这个操作无效。请换一种完全不同的思路")
            history.append(hint)
            log.info("重复动作警告：%s x%d", action_key, _repeat_count)
            _is_loop = _repeat_count >= 5
        elif len(_action_window) >= 6 and len(set(_action_window[-6:])) <= 2:
            cycle = "/".join(sorted(set(_action_window[-6:])))
            hint = (f"⚠ 你在两个动作间反复循环({cycle})，已循环{len(_action_window)//2}轮。"
                    f"这条路走不通，请换一种完全不同的方法——比如用 launch 命令、点其他控件、或 scroll")
            history.append(hint)
            log.info("交替循环警告：%s 窗口=%s", cycle, _action_window[-6:])
            _is_loop = len(_action_window) >= 8 and len(set(_action_window)) <= 2
        if _is_loop:
            # 尝试模型升级自救：flash 卡了 → 临时切 pro 重新决策一次
            llm = getattr(planner, "llm", None)
            cur_model = getattr(llm, "model", "") if llm else ""
            if "flash" in cur_model and not getattr(llm, "_escalated", False):
                pro_model = cur_model.replace("flash", "pro")
                log.info("动作循环 → 临时升级 %s → %s 重试", cur_model, pro_model)
                llm.model = pro_model
                llm._escalated = True  # type: ignore[attr-defined]
                history.append(f"⚠ 框架已临时升级到更强模型({pro_model})，请重新审视屏幕并换策略")
                _action_window.clear()
                _repeat_count = 0
                continue  # 重新 observe + act（用 pro 模型）
            return manual_stop(executor, obs, f"动作循环无法脱困：{_action_window[-4:]}",
                               run_dir=run_dir, image=image)

        if action.kind == "done":
            log.info("✅ 任务完成：%s", task)
            _save_audit(run_dir, f"{_stamp()}_DONE", image, obs, {"task": task})
            return DONE
        if action.kind == "stop":
            return manual_stop(executor, obs, f"agent 停止：{action.reason}", run_dir=run_dir, image=image)

        budget.tick()  # 只对真正执行的动作计预算
        if dry_run:
            log.info("[dry-run] 预览下一步：%s %s | 原因=%s", action.kind, action.text or action.direction, action.reason)
            return DONE

        # 命令直达/启动（能靠指令就不点）
        if action.kind in ("launch", "intent"):
            # 回退策略：app 打开后可能停在任意页(朋友圈/某聊天/文章)。LLM 若想重启一个「已在前台」
            # 的 app，重启只会停回同一页 → 改按「返回」回退到主页；连续多次仍不行才交人工。
            cur_pkg = (obs.foreground or "").split("/", 1)[0]
            if action.kind == "launch" and action.text and cur_pkg == action.text:
                recovery_back += 1
                if recovery_back > 4:
                    return manual_stop(
                        executor, obs, f"{action.text} 反复回退仍未到可操作页面", run_dir=run_dir, image=image
                    )
                executor.back()
                log.info("▶ 执行：back（%s 已在前台，回退到主页 第%d次）", action.text, recovery_back)
                history.append(
                    f"{action.text} 已在前台({(obs.foreground or '').split('/')[-1]})，"
                    f"按返回回退一层找主页（第{recovery_back}次，别再重启）"
                )
                wait_for_change(executor, image, timeout=loop.transition_timeout, threshold=loop.transition_threshold)
                time.sleep(loop.after_click_delay)
                continue
            try:
                if action.kind == "launch":
                    executor.launch_app(action.text)
                    desc = f"launch {action.text}"
                else:
                    executor.start_intent(action.text)
                    desc = f"intent {action.text}"
                log.info("▶ 执行：%s", desc)
            except Exception as e:  # noqa: BLE001
                hint = f"命令失败({action.text})：包名不存在。"
                # 尝试用 pm list 搜正确包名提示 LLM
                if action.kind == "launch":
                    try:
                        # 用最后一段的前4字符做模糊搜索（notes→note 能匹配 notepad）
                        kw = action.text.split(".")[-1][:4].lower() if "." in action.text else action.text[:4].lower()
                        out = executor._shell("pm", "list", "packages", timeout=8) if hasattr(executor, "_shell") else ""
                        candidates = [l.replace("package:", "").strip() for l in out.splitlines()
                                      if kw in l.lower()][:5]
                        if candidates:
                            hint += f" 可能的包名：{', '.join(candidates)}。"
                        else:
                            hint += " 建议 back 到桌面/应用抽屉找图标 tap 打开。"
                    except Exception:  # noqa: BLE001
                        pass
                history.append(hint)
                time.sleep(loop.poll_interval)
                continue
            wait_for_change(executor, image, timeout=loop.transition_timeout, threshold=loop.transition_threshold)
            history.append(f"{desc} -> 已执行（{action.reason}）")
            no_change = 0
            time.sleep(loop.after_click_delay)
            continue

        recovery_back = 0  # 走到这里说明是正常动作(tap/text/scroll…)，清零回退计数
        # 解析执行参数 + 安全 + RiskGate（与 hybrid 一致）
        point: Optional[tuple[int, int]] = None
        target_text = ""
        if action.uses_index:
            try:
                target = validate_action(action, obs.elements, profile.forbidden_markers)
            except SafetyError as e:
                history.append(f"{action.kind} index={action.index} 非法({e})")
                time.sleep(loop.poll_interval)
                continue
            if gate.is_forbidden(target.text):
                return manual_stop(executor, obs, f"目标在黑名单：{target.text}", run_dir=run_dir, image=image)
            point, target_text = target.center, target.text
            if not gate.approve(action, target_text):
                return manual_stop(executor, obs, f"动作被拒绝：{target_text}", run_dir=run_dir, image=image)
        elif action.kind in ("tap", "long_press") and action.point:
            try:
                point = validate_point(action.point, (obs.image_w, obs.image_h))
            except SafetyError as e:
                history.append(f"坐标无效({e})")
                time.sleep(loop.poll_interval)
                continue
            target_text = f"坐标{point}"
            if not gate.approve(action, target_text, is_coordinate=True):
                return manual_stop(executor, obs, "坐标点击被拒绝", run_dir=run_dir, image=image)
        elif action.kind == "text":
            if not gate.approve(action, action.text):
                return manual_stop(executor, obs, "文本输入被拒绝", run_dir=run_dir, image=image)
            executor.prepare_input()
            _focus_input(executor, obs, [], profile, planner=planner, image=image, point=action.point)
            time.sleep(0.6)
        else:
            if action.kind in ("scroll", "fling", "swipe", "drag"):
                if not budget.scroll_ok():
                    return manual_stop(executor, obs, "滚动次数超预算", run_dir=run_dir, image=image)
                budget.scrolls += 1

        _save_audit(run_dir, f"{_stamp()}_act", image, obs,
                    {"mode": "llm", "kind": action.kind, "target": target_text, "reason": action.reason})
        try:
            desc = execute_action(executor, action, point=point, image_wh=(obs.image_w, obs.image_h))
        except SafetyError as e:
            history.append(f"执行失败({e})")
            time.sleep(loop.poll_interval)
            continue
        log.info("▶ 执行：%s", desc)

        changed = None if action.is_nav_gesture else wait_for_change(
            executor, image, timeout=loop.transition_timeout, threshold=loop.transition_threshold
        )
        chg = "变化" if (action.is_nav_gesture or changed is not None) else "无变化"
        history.append(f"{desc} {target_text} -> {chg}（{action.reason}）")
        if chg == "无变化" and not action.is_nav_gesture:
            no_change += 1
            if no_change > budget.max_no_change_retries + 2:
                return manual_stop(executor, obs, "连续多次操作无变化", run_dir=run_dir, image=image)
        else:
            no_change = 0
        time.sleep(loop.after_click_delay)
