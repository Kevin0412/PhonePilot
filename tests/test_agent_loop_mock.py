"""端到端：MockExecutor + MockLLM 跑通整条 感知→决策→执行 闭环。"""

from phonepilot.agent.budget import Budget
from phonepilot.agent.loop import DONE, HANDOVER, LoopConfig, _find_input_element, run_agent
from phonepilot.types import Observation
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


def test_launch_app_by_command_not_icon_tap():
    # 「打开设置」给了 package -> 用命令启动，不点桌面图标
    frames = [
        MockFrame([el("某桌面图标")], "launcher"),
        MockFrame([el("蓝牙"), el("电池"), el("WLAN")], "com.android.settings"),
        MockFrame([el("可用网络")], "settings/wlan"),
    ]
    plan = {
        "steps": [
            {"goal": "打开设置", "success_markers": ["蓝牙", "电池"], "target_markers": ["设置"], "package": "com.android.settings"},
            {"goal": "进入WLAN", "success_markers": ["可用网络"], "target_markers": ["WLAN"]},
        ],
        "final_success_markers": ["可用网络"],
    }
    rc, ex = _run(frames, plan)
    kinds = [a["kind"] for a in ex.actions]
    assert rc == DONE
    assert kinds[0] == "launch" and ex.actions[0]["package"] == "com.android.settings"
    assert kinds.count("tap") == 1  # 只点了 WLAN，没点桌面图标


def test_intent_step_completes_by_navigation_not_text():
    # intent 直达目标页；即便页面文字与计划的 success_markers 不一致(ROM 差异)，也应靠导航判完成
    frames = [
        MockFrame([el("桌面")], "launcher"),
        MockFrame([el("WLAN"), el("可用WLAN"), el("某网络")], "com.android.settings/WifiSettings"),
    ]
    plan = {
        "steps": [{
            "goal": "进入WLAN页面", "success_markers": ["可用网络"], "target_markers": ["WLAN"],
            "intent": "android.settings.WIFI_SETTINGS",
        }],
        "final_success_markers": ["可用网络"],
    }
    rc, ex = _run(frames, plan)
    kinds = [a["kind"] for a in ex.actions]
    assert rc == DONE
    assert kinds == ["intent"]  # 只发 intent，零点击
    assert ex.actions[0]["action"] == "android.settings.WIFI_SETTINGS"


def test_input_text_step_types_via_executor():
    frames = [
        MockFrame([el("发消息")], "chat"),
        MockFrame([el("已输入")], "chat"),
    ]
    plan = {
        "steps": [{"goal": "输入消息", "target_markers": ["发消息"], "input_text": "测试消息"}],
        "final_success_markers": [],
    }
    rc, ex = _run(frames, plan, gate=RiskGate(auto_confirm_medium=True))
    texts = [a for a in ex.actions if a["kind"] == "text"]
    assert rc == DONE
    assert texts and texts[0]["value"] == "测试消息"


def test_replan_when_stuck_then_bounded_handover():
    # 目标永远找不到 -> 卡住 -> 重规划(有上限) -> 最终交人工
    frames = [MockFrame([el("空白页")], "x")]
    plan = {"steps": [{"goal": "找不存在的东西", "target_markers": ["不存在"]}], "final_success_markers": ["不存在"]}
    budget = Budget(max_scrolls=0, max_replans=2, unknown_stop_seconds=0.0)
    rc, _ex = _run(frames, plan, budget=budget)
    assert rc == HANDOVER


def test_verify_blocks_action_on_wrong_contact():
    # 在「沈俊」聊天里，但本步 verify=["王信"] -> 校验失败 -> 不执行发送 -> 交人工（防误发）
    frames = [MockFrame([el("沈俊"), el("发送"), el("测试消息")], "chat")]
    plan = {
        "steps": [{"goal": "发送", "target_markers": ["发送"], "verify": ["王信"], "risk": "high"}],
        "final_success_markers": [],
    }
    gate = RiskGate(high_risk_markers=["发送"], confirm_fn=lambda _p: "y")
    rc, ex = _run(frames, plan, gate=gate, budget=Budget(max_replans=0))
    assert rc == HANDOVER
    assert not any(a["kind"] == "tap" for a in ex.actions)  # 未点击发送


def test_verify_passes_then_acts():
    frames = [
        MockFrame([el("王信"), el("发送")], "chat"),
        MockFrame([el("王信"), el("已发送消息")], "chat"),
    ]
    plan = {
        "steps": [{"goal": "发送", "target_markers": ["发送"], "verify": ["王信"], "success_markers": ["已发送消息"]}],
        "final_success_markers": [],
    }
    rc, ex = _run(frames, plan)
    assert rc == DONE
    assert any(a["kind"] == "tap" for a in ex.actions)  # 校验通过，执行了点击


def test_llm_mode_drives_steps_to_done():
    from phonepilot.agent.loop import run_llm_agent

    frames = [
        MockFrame([el("设置", center=(620, 1300))], "launcher"),
        MockFrame([el("WLAN", center=(540, 520))], "settings"),
        MockFrame([el("可用网络")], "wlan"),
    ]
    ex = MockExecutor(frames)
    ocr = MockOcrEngine(ex)
    llm = MockLLMClient(decisions=[
        {"action": "tap", "index": 0, "text": "设置"},
        {"action": "tap", "index": 0, "text": "WLAN"},
        {"action": "done"},
    ])
    rc = run_llm_agent(
        "打开设置进入WLAN", executor=ex, ocr=ocr, planner=Planner(llm),
        gate=RiskGate(), profile=Profile(), budget=Budget(), loop=_fast_loop(),
    )
    assert rc == DONE
    assert len([a for a in ex.actions if a["kind"] == "tap"]) == 2


def test_relaunch_when_already_foreground_recovers_via_back():
    """回退策略：app 已在前台但停在非预期页(朋友圈)，LLM 误判要重启时，
    循环改按「返回」回退找主页（绝不重启已在前台的 app），多次仍不行才交人工。"""
    from phonepilot.agent.loop import run_llm_agent

    fg = "com.tencent.mm/com.tencent.mm.plugin.sns.ui.SnsBrowseUI"
    frames = [MockFrame([el("朋友圈"), el("相机")], fg)]
    ex = MockExecutor(frames)
    ocr = MockOcrEngine(ex)
    llm = MockLLMClient(decisions=[{"action": "launch", "package": "com.tencent.mm"}] * 8)
    rc = run_llm_agent(
        "打开微信给某人发消息", executor=ex, ocr=ocr, planner=Planner(llm),
        gate=RiskGate(), profile=Profile(), budget=Budget(), loop=_fast_loop(),
    )
    assert rc == HANDOVER
    assert not [a for a in ex.actions if a["kind"] == "launch"]  # 从不重启已在前台的 app
    backs = [a for a in ex.actions if a["kind"] == "key" and "BACK" in str(a.get("keycode", "")).upper()]
    assert len(backs) == 4  # 连续回退上限 4 次后交人工


def test_hard_stop_marker_hands_over():
    # 硬停标志用「滑块/人机验证」这类几乎不会出现在正常内容里的短语；
    # 通用内容词(如"验证码")不再当硬停子串，避免聊天/朋友圈内容误触发（由 LLM 判断+返回 stop 兜底）。
    frames = [
        MockFrame([el("设置")], "launcher"),
        MockFrame([el("拖动滑块完成验证"), el("确定")], "captcha"),
    ]
    plan = {
        "steps": [{"goal": "x", "success_markers": ["永不出现"], "target_markers": ["设置"]}],
        "final_success_markers": ["永不出现"],
    }
    rc, ex = _run(frames, plan)
    assert rc == HANDOVER  # 命中「拖动滑块」硬停止
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


def test_find_input_element_picks_bottom_wide_box():
    """_find_input_element 应找到底部宽矮的 ui/edit 元素作为输入框。"""
    obs = Observation(
        image_w=1080, image_h=2400,
        elements=[
            el("标题", center=(540, 100)),
            el("[输入框]", center=(500, 2300), kind="ui"),  # 底部宽矮框
        ],
        signature="", foreground="",
    )
    # 需要让 bounds 足够宽
    from phonepilot.types import ScreenElement
    wide_input = ScreenElement(text="[输入框]", score=0.5,
                               box=((100, 2250), (900, 2250), (900, 2350), (100, 2350)),
                               style="normal", kind="ui")
    obs2 = Observation(image_w=1080, image_h=2400,
                       elements=[el("标题", center=(540, 100)), wide_input],
                       signature="", foreground="")
    xy = _find_input_element(obs2)
    assert xy is not None
    assert xy[1] > 2000  # y 在底部


def test_find_input_element_returns_none_without_bottom_box():
    """没有底部宽矮框时返回 None。"""
    obs = Observation(image_w=1080, image_h=2400,
                      elements=[el("标题", center=(540, 100))],
                      signature="", foreground="")
    assert _find_input_element(obs) is None
