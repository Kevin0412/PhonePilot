import cv2
import numpy as np

from phonepilot.agent.observe import observe
from phonepilot.executor.mock import MockExecutor, MockFrame, MockOcrEngine
from phonepilot.perception import cv_controls
from tests.conftest import el


def test_detect_boxes_finds_a_drawn_rect():
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 150), (300, 230), (0, 0, 0), 3)
    boxes = cv_controls.detect_boxes(img, min_wh=20)
    assert any(abs(x1 - 100) < 12 and abs(y1 - 150) < 12 for (x1, y1, x2, y2) in boxes)


def test_detect_boxes_blank_image_few():
    img = np.full((400, 400, 3), 245, np.uint8)  # 近白纯色，无对比 -> 框不出
    assert cv_controls.detect_boxes(img) == []


def test_input_bar_center_from_bottom_icon_row():
    # 底部一行图标(有对比) -> 反推输入栏纵向中心；y 自适应取该行、x 取偏左安全值(0.40w)
    h, w = 800, 400
    img = np.full((h, w, 3), 245, np.uint8)
    for cx in (40, 130, 300, 350):  # 语音/表情/+/麦克风 样图标，y≈0.95h
        cv2.rectangle(img, (cx, 745), (cx + 36, 781), (0, 0, 0), -1)
    c = cv_controls.input_bar_center(img)
    assert c is not None
    assert c[0] == int(w * 0.40)          # x 偏左安全值
    assert 0.88 * h <= c[1] <= 0.985 * h  # y 落在底部输入栏带内（自适应得来，非写死比例）


def test_input_bar_center_rejects_when_no_bottom_icons():
    img = np.full((800, 400, 3), 245, np.uint8)  # 纯色无图标 -> 不可信，返回 None 交上层兜底
    assert cv_controls.input_bar_center(img) is None


class _MockDetector:
    """测试用：返回固定元素列表的假检测器。"""
    def __init__(self, elements):
        self._elements = elements
    def detect(self, image):
        return self._elements


def test_observe_ui_detect_replaces_cv():
    """检测器开启时替代粗糙 CV 画框，且与 OCR 去重。"""
    img = np.full((600, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 480), (160, 560), (0, 0, 0), 3)  # 会被粗糙 CV 看到的框
    det_elem = el("[控件]", center=(300, 500), kind="ui", score=0.8)
    detector = _MockDetector([det_elem])
    frame = MockFrame(elements=[el("标题", center=(200, 40))], foreground="app", width=400, height=600, image=img)
    ex = MockExecutor([frame])
    ocr = MockOcrEngine(ex)
    _img, obs = observe(ex, ocr, None, ui_detect=detector, with_foreground=False, with_cv=True)
    ui = [e for e in obs.elements if e.kind == "ui"]
    assert ui  # 检测器的元素在
    cv = [e for e in obs.elements if e.kind == "control"]
    assert not cv  # 粗糙 CV 被跳过了（检测器替代）


def test_observe_cv_fallback_when_no_detector():
    """无检测器时粗糙 CV 仍正常工作。"""
    img = np.full((600, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 480), (160, 560), (0, 0, 0), 3)
    frame = MockFrame(elements=[el("标题", center=(200, 40))], foreground="app", width=400, height=600, image=img)
    ex = MockExecutor([frame])
    ocr = MockOcrEngine(ex)
    _img, obs = observe(ex, ocr, None, ui_detect=None, with_foreground=False, with_cv=True)
    cv = [e for e in obs.elements if e.kind == "control"]
    assert cv  # 粗糙 CV 正常工作


def test_observe_fuses_cv_control_not_overlapping_text():
    img = np.full((600, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 480), (160, 560), (0, 0, 0), 3)  # 底部一个图标样控件
    frame = MockFrame(elements=[el("标题", center=(200, 40))], foreground="app", width=400, height=600, image=img)
    ex = MockExecutor([frame])
    ocr = MockOcrEngine(ex)
    _img, obs = observe(ex, ocr, None, with_foreground=False, with_cv=True)
    controls = [e for e in obs.elements if e.kind == "control"]
    assert controls and any(e.text == "[控件]" for e in controls)
    # 文字元素仍在
    assert any(e.text == "标题" for e in obs.elements)
    # 签名只含文字，不含 CV 控件框（保持稳定）
    assert "标题" in obs.signature and "控件" not in obs.signature
