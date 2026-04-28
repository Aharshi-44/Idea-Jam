from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response

from services.image_generator import ImageGenerator
from services.watermark_encoder import image_to_bytes
from services.watermark_pipeline import apply_semantic_watermark


router = APIRouter()
generator = ImageGenerator()


def _extract_prompt(form_prompt: Optional[str], request_json: object) -> str:
    if form_prompt is not None and form_prompt.strip():
        return form_prompt.strip()

    if isinstance(request_json, dict):
        json_prompt = request_json.get("prompt")
        if isinstance(json_prompt, str) and json_prompt.strip():
            return json_prompt.strip()

    raise HTTPException(status_code=400, detail="A non-empty 'prompt' is required.")


def _is_cuda_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "out of memory" in message and "cuda" in message
    return False


@router.post("")
async def generate_route(request: Request, prompt: Optional[str] = Form(None)) -> Response:
    request_json = {}
    if prompt is None:
        try:
            request_json = await request.json()
        except Exception:
            request_json = {}

    final_prompt = _extract_prompt(prompt, request_json)

    try:
        image = generator.generate(final_prompt)
        image_rgb = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        marked_bgr, _, _ = apply_semantic_watermark(image_bgr)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if _is_cuda_oom_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail="GPU memory error") from exc
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc

    buffer = BytesIO(image_to_bytes(marked_bgr, ext=".png"))
    return Response(content=buffer.getvalue(), media_type="image/png")
