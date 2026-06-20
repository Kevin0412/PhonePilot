from phonepilot.agent.local_rules import local_rule_decision
from phonepilot.profile import Profile
from phonepilot.types import Step
from tests.conftest import el


PROFILE = Profile()


def _step(targets, succ=()):
    return Step(goal="g", success_markers=list(succ), target_markers=list(targets))


def test_target_found_returns_tap():
    els = [el("设置"), el("WLAN")]
    a = local_rule_decision(els, _step(["WLAN"]), PROFILE)
    assert a.kind == "tap" and a.index == 1 and a.text == "WLAN"


def test_recovery_takes_priority_over_target():
    els = [el("网络异常"), el("重试"), el("WLAN")]
    a = local_rule_decision(els, _step(["WLAN"]), PROFILE)
    assert a.kind == "tap" and "recovery" in a.reason and a.text == "重试"


def test_loading_returns_wait():
    els = [el("正在加载"), el("WLAN")]
    # 加载优先于目标
    a = local_rule_decision(els, _step(["不存在目标"]), PROFILE)
    assert a.kind == "wait" and a.reason.startswith("loading")


def test_scroll_to_find_when_target_absent():
    els = [el("显示"), el("声音")]
    a = local_rule_decision(els, _step(["WLAN"]), PROFILE, allow_scroll=True)
    assert a.kind == "scroll" and a.direction == "down"


def test_no_scroll_budget_falls_to_wait():
    els = [el("显示")]
    a = local_rule_decision(els, _step(["WLAN"]), PROFILE, allow_scroll=False)
    assert a.kind == "wait"


def test_no_targets_returns_wait():
    a = local_rule_decision([el("随便")], _step([]), PROFILE)
    assert a.kind == "wait"
