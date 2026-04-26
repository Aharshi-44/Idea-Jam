from fastapi import APIRouter, File, HTTPException, UploadFile
import cv2
import numpy as np

from services.watermark_decoder import detect_watermark


router = APIRouter()


def _decode_upload_to_bgr(payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image upload.")
    return image


@router.post("/detect")
async def detect_route(image: UploadFile = File(...)):
    try:
        payload = await image.read()
        input_bgr = _decode_upload_to_bgr(payload)
        result = detect_watermark(input_bgr)
        return {
            "detected": bool(result["detected"]),
            "confidence": float(result["confidence"]),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

