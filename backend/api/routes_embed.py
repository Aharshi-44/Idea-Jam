from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from services.watermark_encoder import image_to_bytes
from services.watermark_pipeline import apply_semantic_watermark, decode_image_bytes


router = APIRouter()


@router.post("/embed")
async def embed_route(image: UploadFile = File(...)) -> Response:
    try:
        payload = await image.read()
        input_bgr = decode_image_bytes(payload)
        marked, _, _ = apply_semantic_watermark(input_bgr)
        output = image_to_bytes(marked, ext=".png")
        return Response(content=output, media_type="image/png")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

