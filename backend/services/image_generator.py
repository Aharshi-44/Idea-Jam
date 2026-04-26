from __future__ import annotations

from threading import Lock

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class ImageGenerator:
    _instance: "ImageGenerator | None" = None
    _lock = Lock()

    def __new__(cls) -> "ImageGenerator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

        # Memory optimization
        self.pipe.enable_attention_slicing()

        # Move to device
        if self.device == "cuda":
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception:
                    self.pipe = self.pipe.to("cuda")
            else:
                self.pipe = self.pipe.to("cuda")

        # 🔥 Warmup (prevents first-call lag)
        try:
            self.pipe("warmup", num_inference_steps=1)
        except Exception:
            pass

        self._initialized = True

    def generate(self, prompt: str) -> Image.Image:
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        # 🔥 Seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(42)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    result = self.pipe(
                        prompt=cleaned_prompt,
                        num_inference_steps=25,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=generator,
                    )
            else:
                result = self.pipe(
                    prompt=cleaned_prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=generator,
                )

        return result.images[0]