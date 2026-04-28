from typing import Tuple

import cv2
import numpy as np
from fastapi import HTTPException

from services.feature_extractor import FeatureExtractor
from services.semantic_detector import SemanticDetector
from services.watermark_encoder import build_signature, embed_watermark


BBox = Tuple[int, int, int, int]
SECRET_KEY = "hackathon-secret-key"

semantic_detector = SemanticDetector()
feature_extractor = FeatureExtractor()


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image upload.")
    return image


def apply_semantic_watermark(image_bgr: np.ndarray) -> tuple[np.ndarray, BBox, str]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    bbox = semantic_detector.detect_main_region(image_bgr)
    x1, y1, x2, y2 = bbox
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Unable to extract valid ROI.")

    embedding = feature_extractor.extract_embedding(roi)
    signature = build_signature(embedding, SECRET_KEY)
    marked = embed_watermark(image_bgr, bbox, signature)
    return marked, bbox, signature
