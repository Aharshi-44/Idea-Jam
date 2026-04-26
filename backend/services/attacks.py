import cv2
import numpy as np


def attack_crop(image_bgr: np.ndarray, keep_ratio: float = 0.8) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    h, w = image_bgr.shape[:2]
    if h < 2 or w < 2:
        return image_bgr.copy()

    crop_h = max(1, int(h * keep_ratio))
    crop_w = max(1, int(w * keep_ratio))
    y1 = max(0, (h - crop_h) // 2)
    x1 = max(0, (w - crop_w) // 2)
    cropped = image_bgr[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_compress(image_bgr: np.ndarray, jpeg_quality: int = 30) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    quality = int(np.clip(jpeg_quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG compression failed.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decompression failed.")
    return decoded


def attack_noise(image_bgr: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    noise = np.random.normal(0.0, sigma, image_bgr.shape).astype(np.float32)
    noisy = image_bgr.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

