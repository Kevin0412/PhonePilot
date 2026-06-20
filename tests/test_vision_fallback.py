"""视觉兜底：OCR 看不懂的页面升级到视觉模型，返回归一化坐标点击。"""

from phonepilot.agent.budget import Budget
from phonepilot.agent.loop import DONE, HANDOVER, LoopConfig, run_agent
from phonepilot.executor.mock import MockExecutor, MockFrame, MockOcrEngine
from phonepilot.planner.llm_mock import MockLLMClient
from phonepilot.planner.llm_vision import MockVisionClient
from phonepilot.planner.planner import Planner
from phonepilot.profile import Profile
from phonepilot.risk.gate import RiskGate
from tests.conftest import el


def _loop():
    # min_interval 较大避免 worker 重复启动覆盖结果；max_age 较大避免过期
    return LoopConfig(
        poll_interval=0.01, after_click_delay=0.01, transition_timeout=2.0,
        transition_threshold=0.02, llm_max_age=10, vision_max_age=10, llm_min_interval=2.0,
    )


PLAN = {
    "steps": [{"goal": "点击神秘图标", "success_markers": ["可用网络"], "target_markers": ["神秘图标"]}],
    "final_success_markers": ["可用网络"],
}


def _build(confirm):
    # f0 无任何文字（OCR 盲区）-> 触发视觉兜底；f1 出现成功标志
    frames = [
        MockFrame([], "blank_screen"),
        MockFrame([el("可用网络"), el("已连接")], "settings/wlan"),
    ]
    ex = MockExecutor(frames)
    ocr = MockOcrEngine(ex)
    vision = MockVisionClient(default={"action": "tap", "point": [0.5, 0.5], "reason": "看到图标"})
    planner = Planner(MockLLMClient(plan=PLAN), vision)
    gate = RiskGate(confirm_coord_taps=True, confirm_fn=lambda _p: confirm)
    budget = Budget(max_scrolls=0)  # 不许滚动 -> OCR 盲区时本地直接 wait -> 升级视觉
    plan = planner.make_plan("任务")
    rc = run_agent(
        plan, executor=ex, ocr=ocr, planner=planner, gate=gate, profile=Profile(),
        budget=budget, loop=_loop(), vision_enabled=True,
    )
    return rc, ex, vision


def test_vision_tap_completes_when_confirmed():
    rc, ex, vision = _build(confirm="y")
    assert vision.calls >= 1
    assert rc == DONE
    taps = [a for a in ex.actions if a["kind"] == "tap"]
    assert len(taps) == 1  # 视觉给出的坐标点击


def test_vision_tap_declined_hands_over():
    rc, ex, vision = _build(confirm="n")
    assert rc == HANDOVER
    assert len([a for a in ex.actions if a["kind"] == "tap"]) == 0
