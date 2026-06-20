import pytest

from phonepilot.agent.safety import SafetyError, validate_action, validate_point
from phonepilot.types import Action
from tests.conftest import el


def test_validate_action_ok_returns_target():
    els = [el("设置"), el("WLAN")]
    target = validate_action(Action(kind="tap", index=1, text="WLAN"), els)
    assert target.text == "WLAN"


def test_index_missing_raises():
    with pytest.raises(SafetyError):
        validate_action(Action(kind="tap", index=None, text="x"), [el("a")])


def test_index_out_of_range_raises():
    with pytest.raises(SafetyError):
        validate_action(Action(kind="tap", index=5, text="x"), [el("a")])


def test_index_text_mismatch_raises():
    els = [el("设置"), el("WLAN")]
    with pytest.raises(SafetyError):
        validate_action(Action(kind="tap", index=0, text="WLAN"), els)


def test_substring_ambiguity_now_rejected():
    # index 指向「删除」，但 text 是「删除全部」——子串匹配会放过，精确匹配应拒绝
    els = [el("删除"), el("删除全部")]
    with pytest.raises(SafetyError):
        validate_action(Action(kind="tap", index=0, text="删除全部"), els)


def test_blacklisted_target_raises():
    els = [el("确认支付")]
    with pytest.raises(SafetyError):
        validate_action(Action(kind="tap", index=0, text="确认支付"), els, forbidden_markers=["确认支付"])


def test_validate_point_clamps_and_scales():
    assert validate_point((0.5, 0.5), (1000, 2000)) == (500, 1000)
    assert validate_point((2.0, -1.0), (1000, 2000)) == (999, 0)  # 越界裁剪


def test_validate_point_invalid_raises():
    with pytest.raises(SafetyError):
        validate_point(None, (1000, 2000))
    with pytest.raises(SafetyError):
        validate_point(("a", "b"), (1000, 2000))
