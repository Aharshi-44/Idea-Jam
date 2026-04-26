from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


BBox = Tuple[int, int, int, int]


class SemanticDetector:
    def __init__(self, model_name: str = "yolov8n.pt", min_roi_size: int = 32) -> None:
        self.model_name = model_name
        self.min_roi_size = min_roi_size
        self._model: Optional[YOLO] = None

    def _load_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_name)
        return self._model

    def _center_fallback_bbox(self, image_shape: Tuple[int, int, int]) -> BBox:
        h, w = image_shape[:2]
        side = max(self.min_roi_size, int(min(h, w) * 0.6))
        side = min(side, h, w)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return self._clip_and_fix_bbox((x1, y1, x2, y2), w, h)

    def _clip_and_fix_bbox(self, bbox: BBox, width: int, height: int) -> BBox:
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))

        if (x2 - x1) < self.min_roi_size:
            pad = self.min_roi_size - (x2 - x1)
            left = pad // 2
            right = pad - left
            x1 = max(0, x1 - left)
            x2 = min(width, x2 + right)
        if (y2 - y1) < self.min_roi_size:
            pad = self.min_roi_size - (y2 - y1)
            top = pad // 2
            bottom = pad - top
            y1 = max(0, y1 - top)
            y2 = min(height, y2 + bottom)

        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        return x1, y1, x2, y2

    def detect_main_region(self, image_bgr: np.ndarray) -> BBox:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty or invalid.")

        h, w = image_bgr.shape[:2]
        if h < 2 or w < 2:
            return 0, 0, max(w, 1), max(h, 1)

        fallback_bbox = self._center_fallback_bbox(image_bgr.shape)
        try:
            model = self._load_model()
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = model.predict(source=rgb_image, verbose=False)
            if not results:
                return fallback_bbox

            first_result = results[0]
            if first_result.boxes is None or len(first_result.boxes) == 0:
                return fallback_bbox

            boxes_xyxy = first_result.boxes.xyxy.cpu().numpy()
            scores = first_result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(scores))
            x1, y1, x2, y2 = boxes_xyxy[best_idx].tolist()
            return self._clip_and_fix_bbox((int(x1), int(y1), int(x2), int(y2)), w, h)
        except Exception:
            return fallback_bbox

