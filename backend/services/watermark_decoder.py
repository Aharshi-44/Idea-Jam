from typing import Dict, List, Tuple

import cv2
import numpy as np

from services.feature_extractor import FeatureExtractor
from services.semantic_detector import SemanticDetector
from services.watermark_encoder import (
    BIT_LENGTH,
    _low_frequency_positions,
    build_signature,
    signature_to_bits,
    watermark_carrier_bboxes,
)


SECRET_KEY = "hackathon-secret-key"
semantic_detector = SemanticDetector()
feature_extractor = FeatureExtractor()


def _extract_votes(
    dct_coeff: np.ndarray,
    bit_count: int,
    redundancy: int,
) -> Tuple[List[int], List[float]]:
    h, w = dct_coeff.shape
    positions = _low_frequency_positions(h, w, bit_count, redundancy)

    bits: List[int] = []
    margins: List[float] = []
    for i in range(bit_count):
        values = []
        for rep in range(redundancy):
            y, x = positions[i * redundancy + rep]
            values.append(float(dct_coeff[y, x]))
        mean_val = float(np.mean(values))
        vote = 1 if mean_val > 0 else 0
        margin = min(1.0, abs(mean_val) / 10.0)
        bits.append(vote)
        margins.append(margin)
    return bits, margins


def _extract_bits_from_bbox(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    redundancy: int,
) -> Tuple[List[int], List[float]]:
    x1, y1, x2, y2 = bbox
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("ROI is empty after clipping.")

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    dct = cv2.dct(luminance)
    return _extract_votes(dct, BIT_LENGTH, redundancy=redundancy)


def detect_watermark(
    image_bgr: np.ndarray,
    redundancy: int = 3,
    confidence_threshold: float = 0.62,
) -> Dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    bbox = semantic_detector.detect_main_region(image_bgr)
    x1, y1, x2, y2 = bbox
    semantic_roi = image_bgr[y1:y2, x1:x2]
    if semantic_roi.size == 0:
        raise ValueError("Unable to extract valid ROI.")

    embedding = feature_extractor.extract_embedding(semantic_roi)
    expected_bits = signature_to_bits(build_signature(embedding, SECRET_KEY))

    carrier_scores: List[float] = []
    for carrier_bbox in watermark_carrier_bboxes(image_bgr.shape, bbox):
        extracted_bits, margins = _extract_bits_from_bbox(image_bgr, carrier_bbox, redundancy=redundancy)
        bit_match = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(extracted_bits, expected_bits)]))
        mean_margin = float(np.mean(margins))
        carrier_scores.append(0.7 * bit_match + 0.3 * mean_margin)

    if not carrier_scores:
        return {"detected": False, "confidence": 0.0}

    strongest_scores = sorted(carrier_scores, reverse=True)[: min(3, len(carrier_scores))]
    confidence = float(np.clip(np.mean(strongest_scores), 0.0, 1.0))
    detected = confidence >= confidence_threshold

    return {
        "detected": bool(detected),
        "confidence": confidence,
    }

