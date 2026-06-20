"""端到端：MockExecutor + MockLLM 跑通整条 感知→决策→执行 闭环。"""

from phonepilot.agent.budget import Budget
from phonepilot.agent.loop import DONE, HANDOVER, LoopConfig, run_agent
from phonepilot.executor.mock import MockExecutor, MockFrame, MockOcrEngine
from phonepilot.planner.llm_mock import MockLLMClient
from phonepilot.planner.planner import Planner
from phonepilot.profile import Profile
from phonepilot.risk.gate import RiskGate
from tests.conftest import el


def _fast_loop():
    return LoopConfig(
        poll_interval=0.005, after_click_delay=0.005, transition_timeout=1.5,
        transition_threshold=0.02, llm_max_age=5, vision_max_age=5, llm_min_interval=0.0,
    )


def _run(frames, plan_json, *, gate=None, profile=None, budget=None):
    ex = MockExecutor(frames)
    ocr = MockOcrEngine(ex)
    planner = Planner(MockLLMClient(plan=plan_json))
    plan = planner.make_plan("任务")
    rc = run_agent(
        plan, executor=ex, ocr=ocr, planner=planner,
        gate=gate or RiskGate(auto_confirm_low=True),
        profile=profile or Profile(), budget=budget or Budget(), loop=_fast_loop(),
    )
    return rc, ex


DEMO_PLAN = {
    "goal": "进入 WLAN",
    "steps": [
        {"goal": "打开设置", "success_markers": ["WLAN", "蓝牙"], "target_markers": ["设置"]},
        {"goal": "进入 WLAN", "success_markers": ["可用网络"], "target_markers": ["WLAN"]},
    ],
    "final_success_markers": ["可用网络"],
    "overall_risk": "low",
}


def test_demo_completes_with_two_taps():
    frames = [
        MockFrame([el("设置", center=(620, 1300)), el("时钟")], "launcher"),
        MockFrame([el("WLAN"), el("蓝牙"), el("显示")], "com.android.settings"),
        MockFrame([el("可用网络"), el("已连接")], "settings/wlan"),
    ]
    rc, ex = _run(frames, DEMO_PLAN)
    taps = [a for a in ex.actions if a["kind"] == "tap"]
    assert rc == DONE
    assert len(taps) == 2


def test_hard_stop_marker_hands_over():
    frames = [
        MockFrame([el("设置")], "launcher"),
        MockFrame([el("请输入验证码"), el("确定")], "captcha"),
    ]
    plan = {
        "steps": [{"goal": "x", "success_markers": ["永不出现"], "target_markers": ["设置"]}],
        "final_success_markers": ["永不出现"],
    }
    rc, ex = _run(frames, plan)
    assert rc == HANDOVER  # 命中「验证码」硬停止
    assert len([a for a in ex.actions if a["kind"] == "tap"]) == 1


def test_high_risk_declined_hands_over():
    frames = [MockFrame([el("删除账户"), el("取消")], "app")]
    plan = {
        "steps": [{"goal": "删账户", "success_markers": ["永不出现"], "target_markers": ["删除账户"], "risk": "high"}],
        "final_success_markers": ["永不出现"],
    }
    gate = RiskGate(high_risk_markers=["删除"], confirm_fn=lambda _p: "n")
    rc, ex = _run(frames, plan, gate=gate)
    assert rc == HANDOVER
    assert len([a for a in ex.actions if a["kind"] == "tap"]) == 0  # 被拒，未点击


def test_high_risk_confirmed_executes():
    frames = [
        MockFrame([el("删除账户")], "app"),
        MockFrame([el("已删除")], "app"),
    ]
    plan = {
        "steps": [{"goal": "删账户", "success_markers": ["已删除"], "target_markers": ["删除账户"], "risk": "high"}],
        "final_success_markers": ["已删除"],
    }
    gate = RiskGate(high_risk_markers=["删除"], confirm_fn=lambda _p: "y")
    rc, ex = _run(frames, plan, gate=gate)
    assert rc == DONE
    assert len([a for a in ex.actions if a["kind"] == "tap"]) == 1
