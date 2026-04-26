from typing import Dict, List, Tuple

import cv2
import numpy as np

from services.watermark_encoder import BIT_LENGTH, _low_frequency_positions


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


def detect_watermark(
    image_bgr: np.ndarray,
    redundancy: int = 3,
    confidence_threshold: float = 0.55,
) -> Dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))

    _, margins = _extract_votes(dct, BIT_LENGTH, redundancy=redundancy)
    confidence = float(np.clip(np.mean(margins), 0.0, 1.0))
    detected = confidence >= confidence_threshold

    return {
        "detected": bool(detected),
        "confidence": confidence,
    }

