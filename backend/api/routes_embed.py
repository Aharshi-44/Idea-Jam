from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np

from services.feature_extractor import FeatureExtractor
from services.semantic_detector import SemanticDetector
from services.watermark_encoder import build_signature, embed_watermark, image_to_bytes


router = APIRouter()
semantic_detector = SemanticDetector()
feature_extractor = FeatureExtractor()
SECRET_KEY = "hackathon-secret-key"


def _decode_upload_to_bgr(upload: UploadFile, payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image upload.")
    return image


@router.post("/embed")
async def embed_route(image: UploadFile = File(...)) -> Response:
    try:
        payload = await image.read()
        input_bgr = _decode_upload_to_bgr(image, payload)
        display_bgr = input_bgr.copy()  

        bbox = semantic_detector.detect_main_region(input_bgr)
        x1, y1, x2, y2 = bbox
        roi = input_bgr[y1:y2, x1:x2]

        overlay = display_bgr.copy()

# Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)

# Blend overlay with original image
        alpha = 0.2  # transparency level
        display_bgr = cv2.addWeighted(overlay, alpha, display_bgr, 1 - alpha, 0)

# Optional border for clarity
        cv2.rectangle(display_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Optional label
        cv2.putText(display_bgr,"Semantic Region",(x1, max(y1 - 10, 0)),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)
        if roi.size == 0:
            raise HTTPException(status_code=400, detail="Unable to extract valid ROI.")

        embedding = feature_extractor.extract_embedding(roi)
        signature = build_signature(embedding, SECRET_KEY)
        marked = embed_watermark(display_bgr, bbox, signature)
        output = image_to_bytes(marked, ext=".png")
        return Response(content=output, media_type="image/png")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

