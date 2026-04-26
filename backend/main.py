from fastapi import FastAPI

from api.routes_attack import router as attack_router
from api.routes_detect import router as detect_router
from api.routes_embed import router as embed_router
from api.routes_generate import router as generate_router


app = FastAPI(title="Semantic-Aware Resilient Watermarking API")

app.include_router(embed_router)
app.include_router(detect_router)
app.include_router(attack_router)
app.include_router(generate_router, prefix="/generate")


@app.get("/health")
def health():
    return {"status": "ok"}

