import hashlib
from typing import List, Tuple

import cv2
import numpy as np


BIT_LENGTH = 64  # 16 hex chars = 64 bits


def image_to_bytes(image_bgr: np.ndarray, ext: str = ".png") -> bytes:
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError("Failed to encode image.")
    return encoded.tobytes()


def build_signature(embedding: np.ndarray, secret_key: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(embedding.astype(np.float32).tobytes())
    hasher.update(secret_key.encode("utf-8"))
    return hasher.hexdigest()[:16]


def signature_to_bits(signature: str) -> List[int]:
    raw = bytes.fromhex(signature)
    bits: List[int] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _low_frequency_positions(
    height: int,
    width: int,
    bit_count: int,
    redundancy: int,
    offset: int = 4,
) -> List[Tuple[int, int]]:
    usable_h = max(1, min(height - offset, 24))
    usable_w = max(1, min(width - offset, 24))
    positions: List[Tuple[int, int]] = []
    y = offset
    x = offset
    for _ in range(bit_count * redundancy):
        positions.append((y, x))
        x += 1
        if x >= usable_w + offset:
            x = offset
            y += 1
            if y >= usable_h + offset:
                y = offset
    return positions


def _embed_bits_in_dct(
    dct_coeff: np.ndarray,
    bits: List[int],
    alpha: float,
    redundancy: int,
) -> np.ndarray:
    h, w = dct_coeff.shape
    positions = _low_frequency_positions(h, w, len(bits), redundancy)
    for i, bit in enumerate(bits):
        for rep in range(redundancy):
            y, x = positions[i * redundancy + rep]
            delta = alpha if bit == 1 else -alpha
            dct_coeff[y, x] += delta
    return dct_coeff


def embed_watermark(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    signature: str,
    alpha: float = 3.0,
    redundancy: int = 3,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1 = int(np.clip(x1, 0, max(w - 1, 0)))
    y1 = int(np.clip(y1, 0, max(h - 1, 0)))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("ROI is empty after clipping.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_f32 = np.float32(gray)
    dct = cv2.dct(gray_f32)

    bits = signature_to_bits(signature)
    dct_marked = _embed_bits_in_dct(dct, bits, alpha=alpha, redundancy=redundancy)

    recovered = cv2.idct(dct_marked)
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    recovered_bgr = cv2.cvtColor(recovered, cv2.COLOR_GRAY2BGR)

    output = image_bgr.copy()
    output[y1:y2, x1:x2] = recovered_bgr
    return output

