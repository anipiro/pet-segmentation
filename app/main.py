"""HTTP API + web demo for pet segmentation / background removal.

    uvicorn app.main:app --port 8000          # then open http://localhost:8000

Endpoints
  GET  /health            model info
  POST /predict           multipart "file" -> JSON: stats + base64 PNGs (mask, overlay, cutout)
  POST /cutout            multipart "file" -> image/png with transparent background

Model path comes from PETSEG_MODEL (default: models/petseg.onnx, FP32 at 224 px;
models/petseg_int8.onnx is the 3.5x smaller INT8 variant).
"""
import base64
import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from petseg.inference import Predictor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("PETSEG_MODEL", ROOT / "models" / "petseg.onnx"))
MAX_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2048  # larger uploads are downsized first; the network runs at a fixed size anyway

app = FastAPI(title="Pet Segmentation API", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
predictor = Predictor(MODEL_PATH, threads=int(os.environ.get("PETSEG_THREADS", 4)))


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 15 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "could not decode image")
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "input_size": predictor.size,
            "threshold": predictor.threshold}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = await read_image(file)
    t0 = time.perf_counter()
    p = predictor.predict(img)
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "width": img.width,
        "height": img.height,
        "foreground_ratio": round(p.foreground_ratio, 4),
        "pet_found": p.bbox() is not None and p.foreground_ratio > 0.005,
        "bbox": p.bbox(),
        "model_ms": round(p.latency_ms, 1),
        "total_ms": round(total_ms, 1),
        "mask_png": png_b64(p.mask_image()),
        "overlay_png": png_b64(p.overlay()),
        "cutout_png": png_b64(p.cutout()),
    }


@app.post("/cutout", response_class=Response)
async def cutout(file: UploadFile = File(...), crop: bool = Query(True)):
    p = predictor.predict(await read_image(file))
    buf = io.BytesIO()
    p.cutout(crop=crop).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
