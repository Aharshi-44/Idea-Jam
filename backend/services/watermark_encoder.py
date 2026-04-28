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
    vector = np.asarray(embedding, dtype=np.float32).flatten()
    if vector.size == 0:
        raise ValueError("Embedding is empty or invalid.")

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm

    sampled_indices = np.linspace(0, vector.size - 1, BIT_LENGTH, dtype=int)
    sampled_values = vector[sampled_indices]
    adaptive_threshold = float(np.mean(sampled_values) + (len(secret_key) % 3 - 1) * 0.01)
    bits = [1 if value >= adaptive_threshold else 0 for value in sampled_values]

    packed = bytearray()
    for start in range(0, BIT_LENGTH, 8):
        byte_value = 0
        for bit in bits[start : start + 8]:
            byte_value = (byte_value << 1) | bit
        packed.append(byte_value)
    return packed.hex()


def signature_to_bits(signature: str) -> List[int]:
    raw = bytes.fromhex(signature)
    bits: List[int] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _clip_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, max(width - 1, 0)))
    y1 = int(np.clip(y1, 0, max(height - 1, 0)))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def watermark_carrier_bboxes(
    image_shape: Tuple[int, int, int],
    base_bbox: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = _clip_bbox(base_bbox, width, height)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0

    candidates = [
        (x1, y1, x2, y2),
    ]

    layouts = [
        (0.84, 0.84, 0.0, 0.0),
        (0.82, 0.82, -0.08, 0.0),
        (0.82, 0.82, 0.08, 0.0),
        (0.82, 0.82, 0.0, -0.08),
        (0.82, 0.82, 0.0, 0.08),
    ]
    for scale_x, scale_y, shift_x, shift_y in layouts:
        half_width = box_width * scale_x / 2.0
        half_height = box_height * scale_y / 2.0
        shifted_center_x = center_x + box_width * shift_x
        shifted_center_y = center_y + box_height * shift_y
        candidate = (
            int(round(shifted_center_x - half_width)),
            int(round(shifted_center_y - half_height)),
            int(round(shifted_center_x + half_width)),
            int(round(shifted_center_y + half_height)),
        )
        candidates.append(_clip_bbox(candidate, width, height))

    unique_boxes: List[Tuple[int, int, int, int]] = []
    seen = set()
    for candidate in candidates:
        clipped = _clip_bbox(candidate, width, height)
        if clipped in seen:
            continue
        cx1, cy1, cx2, cy2 = clipped
        if (cx2 - cx1) < 32 or (cy2 - cy1) < 32:
            continue
        seen.add(clipped)
        unique_boxes.append(clipped)
    return unique_boxes


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

    bits = signature_to_bits(signature)
    output = image_bgr.copy()

    for x1, y1, x2, y2 in watermark_carrier_bboxes(output.shape, bbox):
        roi = output[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        luminance = ycrcb[:, :, 0].astype(np.float32)
        dct = cv2.dct(luminance)
        dct_marked = _embed_bits_in_dct(dct, bits, alpha=alpha, redundancy=redundancy)

        recovered_luminance = cv2.idct(dct_marked)
        ycrcb[:, :, 0] = np.clip(recovered_luminance, 0, 255).astype(np.uint8)
        recovered_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        output[y1:y2, x1:x2] = recovered_bgr

    return output

