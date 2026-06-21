"""基于 ONNX 的通用 UI 控件检测器（OmniParser icon_detect 等 YOLO 模型）。

输入截图，输出可交互控件的包围盒列表（按钮/输入框/图标等），补 OCR+模板图标看不到的。
重型依赖(onnxruntime/cv2/numpy)全部懒加载——import 本模块不拉起任何重型库。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..types import ScreenElement
from .base import UiDetector

log = logging.getLogger(__name__)


class OmniParserDetector(UiDetector):
    """用 OmniParser icon_detect ONNX 检测 UI 控件。

    模型输入 dynamic shape ``[1,3,H,W]``，输出 ``[1,5,N]``（cx,cy,w,h,score）。
    long_side 越大越精细但越慢（1280 ≈ 41ms CPU/帧，640 ≈ 13ms）。
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.15,
        iou_threshold: float = 0.45,
        long_side: int = 1280,
        providers: Optional[list[str]] = None,
    ) -> None:
        self._path = Path(model_path)
        self._confidence = confidence
        self._iou = iou_threshold
        self._long_side = long_side
        self._providers = providers
        self._session: Any = None  # 懒加载

    def _load(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort

        provs = self._providers or ort.get_available_providers()
        self._session = ort.InferenceSession(str(self._path), providers=provs)
        log.info("UI 检测模型加载完成：%s (%s)", self._path.name, provs[0])
        return self._session

    def detect(self, image: Any) -> list[ScreenElement]:
        import cv2
        import numpy as np

        sess = self._load()
        h, w = image.shape[:2]
        lb, scale = self._letterbox(image)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        out = sess.run(None, {"images": blob})[0]  # [1,5,N]
        preds = out[0].T  # [N,5]: cx,cy,w,h,score
        keep = preds[:, 4] > self._confidence
        preds = preds[keep]
        if len(preds) == 0:
            return []

        boxes = np.zeros((len(preds), 4), np.float32)
        boxes[:, 0] = (preds[:, 0] - preds[:, 2] / 2) / scale
        boxes[:, 1] = (preds[:, 1] - preds[:, 3] / 2) / scale
        boxes[:, 2] = (preds[:, 0] + preds[:, 2] / 2) / scale
        boxes[:, 3] = (preds[:, 1] + preds[:, 3] / 2) / scale
        scores = preds[:, 4]

        # 按面积分层 NMS：大框(如输入栏)包含多个小框(图标)时，单类 NMS 会让高 conf
        # 小图标抑制掉低 conf 大输入框。把面积 > 全图 2% 的大框单独保留，小框正常 NMS。
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        big = areas > (w * h * 0.02)
        big_idx = np.where(big)[0]
        small_idx = np.where(~big)[0]
        kept_big = cv2.dnn.NMSBoxes(
            boxes[big_idx].tolist(), scores[big_idx].tolist(), self._confidence, self._iou
        ) if len(big_idx) else []
        kept_big = big_idx[np.array(kept_big).flatten()] if len(kept_big) else np.array([], dtype=int)
        kept_small = cv2.dnn.NMSBoxes(
            boxes[small_idx].tolist(), scores[small_idx].tolist(), self._confidence, self._iou
        ) if len(small_idx) else []
        kept_small = small_idx[np.array(kept_small).flatten()] if len(kept_small) else np.array([], dtype=int)
        idx = np.concatenate([kept_big, kept_small]).astype(int)

        elements: list[ScreenElement] = []
        for i in idx:
            x1, y1, x2, y2 = (int(max(0, v)) for v in boxes[i])
            x2, y2 = min(x2, w), min(y2, h)
            if x2 <= x1 or y2 <= y1:
                continue
            poly = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
            elements.append(ScreenElement(
                text="[控件]",
                score=float(scores[i]),
                box=poly,
                style="normal",
                kind="ui",
            ))
        return elements

    def _letterbox(self, image: Any) -> tuple[Any, float]:
        import cv2
        import numpy as np

        h, w = image.shape[:2]
        scale = self._long_side / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        stride = 32
        ph = (stride - nh % stride) % stride
        pw = (stride - nw % stride) % stride
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((nh + ph, nw + pw, 3), 114, np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale
