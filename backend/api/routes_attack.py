from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np

from services.attacks import attack_compress, attack_crop, attack_noise
from services.watermark_encoder import image_to_bytes


router = APIRouter()


def _decode_upload_to_bgr(payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image upload.")
    return image


@router.post("/attack")
async def attack_route(
    image: UploadFile = File(...),
    type: str = Query(..., pattern="^(crop|compress|noise)$"),
) -> Response:
    try:
        payload = await image.read()
        input_bgr = _decode_upload_to_bgr(payload)

        if type == "crop":
            output_bgr = attack_crop(input_bgr)
        elif type == "compress":
            output_bgr = attack_compress(input_bgr, jpeg_quality=30)
        else:
            output_bgr = attack_noise(input_bgr)

        output = image_to_bytes(output_bgr, ext=".png")
        return Response(content=output, media_type="image/png")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Attack failed: {exc}") from exc

