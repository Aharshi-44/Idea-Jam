

## What We Have Built So Far

- FastAPI backend with modular route structure.
- Semantic region detection using YOLO (`ultralytics`).
- Feature extraction using `torchvision` ResNet18 embeddings.
- Watermark signature generation from ROI embedding + secret key.
- DCT-domain watermark embedding inside the detected ROI.
- Watermark detection endpoint returning detection status and confidence.
- Attack simulation endpoints (`crop`, `compress`, `noise`) to test robustness.
- Text-to-image generation endpoint using Stable Diffusion (`diffusers`).
- Basic health endpoint for service checks.

## Backend Structure

- `backend/main.py` - FastAPI app entrypoint and router wiring.
- `backend/api/routes_embed.py` - `/embed` endpoint.
- `backend/api/routes_detect.py` - `/detect` endpoint.
- `backend/api/routes_attack.py` - `/attack` endpoint.
- `backend/api/routes_generate.py` - `/generate` endpoint.
- `backend/services/` - core watermarking, semantic detection, attacks, and generation logic.

## Requirements

Install dependencies from:

- `backend/requirements.txt`

## Run Locally

From the project root:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

In a second terminal (from project root) start the frontend:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Default frontend proxy target is `http://127.0.0.1:8000` via `BACKEND_BASE_URL`.

## API Endpoints

- `GET /health` - service health check.
- `POST /embed` - upload an image and embed watermark.
- `POST /detect` - upload an image and detect watermark confidence.
- `POST /attack?type=crop|compress|noise` - apply image attack.
- `POST /generate` - generate image from prompt (form or JSON prompt).  

## Current Pipeline

1. Detect semantic ROI in image.
2. Extract deep embedding from ROI.
3. Build deterministic signature from embedding + secret key.
4. Convert signature to bits and embed in low-frequency DCT coefficients.
5. Apply distortions via attack endpoints.
6. Detect watermark by reading DCT coefficient vote margins.

## Notes

- The generation pipeline automatically uses GPU when available.
- YOLO and diffusion model weights are downloaded on first run.
- Current code uses an in-code secret key placeholder for prototype stage.
- Frontend uses Next.js API routes as a proxy layer, so backend code and routes remain unchanged.

