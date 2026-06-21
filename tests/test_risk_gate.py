from phonepilot.risk.gate import RiskGate
from phonepilot.types import Action, RiskLevel


def _gate(answer="n", **kw):
    return RiskGate(high_risk_markers=["支付", "删除"], forbidden_markers=["确认支付"], confirm_fn=lambda _p: answer, **kw)


def test_is_forbidden():
    assert _gate().is_forbidden("确认支付")
    assert not _gate().is_forbidden("立即购买")


def test_classify_marker_bump_to_high():
    g = _gate()
    assert g.classify(Action(kind="tap", text="立即支付"), "立即支付") == RiskLevel.HIGH


def test_classify_text_input_is_at_least_medium():
    g = _gate()
    assert g.classify(Action(kind="text", text="hello"), "") == RiskLevel.MEDIUM


def test_classify_forbidden_marker_returns_high():
    assert _gate().classify(Action(kind="tap", text="确认支付"), "确认支付") == RiskLevel.HIGH


def test_low_element_tap_auto_approved():
    assert _gate().approve(Action(kind="tap", index=0, text="设置"), "设置") is True


def test_high_requires_confirmation():
    assert _gate(answer="n").approve(Action(kind="tap", text="支付"), "支付") is False
    assert _gate(answer="y").approve(Action(kind="tap", text="支付"), "支付") is True


def test_coordinate_tap_requires_confirmation():
    g = _gate(answer="n")
    assert g.approve(Action(kind="tap", point=(0.5, 0.5)), "<坐标>", is_coordinate=True) is False


def test_medium_auto_confirm_lets_typing_through_but_gates_send():
    g = RiskGate(high_risk_markers=["发送"], auto_confirm_medium=True, confirm_fn=lambda _p: "n")
    assert g.approve(Action(kind="text", text="hi"), "") is True          # 文本输入(medium)放行
    assert g.approve(Action(kind="tap", text="发送"), "发送") is False     # 发送(high)仍需确认


def test_auto_confirm_all_skips_everything():
    g = _gate(answer="n", auto_confirm_all=True)
    assert g.approve(Action(kind="tap", text="支付"), "支付") is True


def test_low_not_auto_when_disabled():
    g = _gate(answer="n", auto_confirm_low=False)
    assert g.approve(Action(kind="tap", index=0, text="设置"), "设置") is False
