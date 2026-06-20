from phonepilot.agent.gestures import execute_action
from phonepilot.types import Action


class RecordExec:
    """记录被调用的执行器原语。"""

    def __init__(self):
        self.calls = []

    def tap(self, x, y):
        self.calls.append(("tap", x, y))

    def swipe(self, x1, y1, x2, y2, ms=300):
        self.calls.append(("swipe", x1, y1, x2, y2, ms))

    def long_press(self, x, y, ms=700):
        self.calls.append(("long_press", x, y, ms))

    def text(self, v):
        self.calls.append(("text", v))

    def key(self, code):
        self.calls.append(("key", code))

    def back(self):
        self.calls.append(("key", "KEYCODE_BACK"))

    def home(self):
        self.calls.append(("key", "KEYCODE_HOME"))


WH = (1080, 2400)


def test_tap_uses_point():
    ex = RecordExec()
    execute_action(ex, Action(kind="tap"), point=(100, 200), image_wh=WH)
    assert ex.calls == [("tap", 100, 200)]


def test_long_press_maps_to_swipe_same_point():
    ex = RecordExec()
    execute_action(ex, Action(kind="long_press"), point=(100, 200), image_wh=WH)
    assert ex.calls[0][0] == "long_press" and ex.calls[0][1:3] == (100, 200)


def test_scroll_down_is_upward_swipe():
    ex = RecordExec()
    execute_action(ex, Action(kind="scroll", direction="down"), point=None, image_wh=WH)
    kind, x1, y1, x2, y2, ms = ex.calls[0]
    assert kind == "swipe" and y1 > y2  # 内容向下看 = 手指上滑


def test_swipe_uses_from_to_norm():
    ex = RecordExec()
    execute_action(ex, Action(kind="swipe", from_norm=(0.1, 0.5), to_norm=(0.9, 0.5)), point=None, image_wh=WH)
    kind, x1, y1, x2, y2, ms = ex.calls[0]
    assert kind == "swipe" and x1 < x2


def test_back_home_text_key():
    ex = RecordExec()
    execute_action(ex, Action(kind="back"), point=None, image_wh=WH)
    execute_action(ex, Action(kind="home"), point=None, image_wh=WH)
    execute_action(ex, Action(kind="text", text="abc"), point=None, image_wh=WH)
    execute_action(ex, Action(kind="key", key="KEYCODE_ENTER"), point=None, image_wh=WH)
    kinds = [c[0] for c in ex.calls]
    assert kinds == ["key", "key", "text", "key"]
    assert ex.calls[2] == ("text", "abc")
