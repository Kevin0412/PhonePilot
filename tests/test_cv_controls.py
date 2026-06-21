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
