import pytest

from phonepilot.planner.llm_base import extract_json
from phonepilot.planner.llm_mock import MockLLMClient
from phonepilot.planner.planner import PlanError, Planner, action_from_dict


# ---- extract_json ---- #
def test_extract_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_with_surrounding_noise():
    assert extract_json('好的：{"action":"tap","index":2} 完成') == {"action": "tap", "index": 2}


def test_extract_nested_braces():
    assert extract_json('{"x":{"y":1},"z":2}') == {"x": {"y": 1}, "z": 2}


def test_extract_failure_raises():
    with pytest.raises(ValueError):
        extract_json("没有 JSON 在这里")


# ---- action_from_dict ---- #
def test_action_alias_click_to_tap():
    a = action_from_dict({"action": "click", "index": "3", "text": "WLAN"})
    assert a.kind == "tap" and a.index == 3 and a.uses_index


def test_action_direction_suffix_inferred():
    a = action_from_dict({"action": "scroll_down"})
    assert a.kind == "scroll" and a.direction == "down"


def test_action_text_input_field():
    a = action_from_dict({"action": "text", "input": "hello"})
    assert a.kind == "text" and a.text == "hello"


def test_action_coordinate_tap():
    a = action_from_dict({"action": "tap", "point": [0.5, 0.8]})
    assert a.kind == "tap" and a.point == (0.5, 0.8) and not a.uses_index


def test_action_illegal_kind_falls_to_wait():
    assert action_from_dict({"action": "explode"}).kind == "wait"


def test_action_non_dict_is_wait():
    assert action_from_dict("garbage").kind == "wait"


def test_tap_without_index_or_point_degrades_to_wait():
    # 防止执行层因「缺少坐标」抛错崩溃
    assert action_from_dict({"action": "tap"}).kind == "wait"
    assert action_from_dict({"action": "long_press", "index": None}).kind == "wait"
    # 有 index 或 point 则保留
    assert action_from_dict({"action": "tap", "index": 0, "text": "x"}).kind == "tap"
    assert action_from_dict({"action": "tap", "point": [0.5, 0.5]}).kind == "tap"


# ---- Planner.make_plan ---- #
def _plan_json():
    return {
        "goal": "g",
        "steps": [{"goal": "打开设置", "success_markers": ["WLAN"], "target_markers": ["设置"], "risk": "low"}],
        "final_success_markers": ["WLAN"],
        "overall_risk": "low",
    }


def test_make_plan_ok():
    p = Planner(MockLLMClient(plan=_plan_json()))
    plan = p.make_plan("打开设置")
    assert len(plan.steps) == 1 and plan.steps[0].target_markers == ["设置"]


def test_make_plan_empty_steps_raises():
    p = Planner(MockLLMClient(plan={"steps": []}))
    with pytest.raises(PlanError):
        p.make_plan("空计划")


def test_make_plan_illegal_risk_defaults_low():
    j = _plan_json()
    j["overall_risk"] = "超级危险"
    plan = Planner(MockLLMClient(plan=j)).make_plan("x")
    assert plan.overall_risk.value == "low"
