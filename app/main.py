"""
app/main.py — REST API untuk model "Prediksi Harga Rumah Jabodetabek".

Jalankan lokal (dari folder project, venv aktif):
  uvicorn app.main:app --reload --port 8000
Lalu buka http://localhost:8000/docs (Swagger UI) untuk mencoba endpoint.

Konfigurasi lewat environment variable:
  MODEL_PATH     path ke model.joblib   (default: model/model.joblib)
  METADATA_PATH  path ke metadata.json  (default: folder yang sama dgn model)
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import (
    BatchRequest,
    BatchResponse,
    HouseFeatures,
    PredictionResponse,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model.joblib"))
METADATA_PATH = Path(
    os.getenv("METADATA_PATH", MODEL_PATH.with_name("metadata.json"))
)

log = logging.getLogger("uvicorn.error")
ml: dict = {}  # diisi SEKALI saat startup, dipakai ulang oleh setiap request


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load model sekali saja (bukan di setiap request!) ---
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH} tidak ada. Jalankan train.py dulu")
    ml["model"] = joblib.load(MODEL_PATH)
    meta = {}
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
    ml["metadata"] = meta
    ml["version"] = meta.get("model_version", "unknown")
    log.info("Model %s dimuat dari %s", ml["version"], MODEL_PATH)
    yield
    # --- shutdown: bersihkan ---
    ml.clear()


app = FastAPI(
    title="Harga Rumah API",
    description="Prediksi harga rumah Jabodetabek — Day 46 DSML Batch 43",
    version="1.0.0",
    lifespan=lifespan,
)


def predict_harga(items: list[HouseFeatures]) -> list[int]:
    """Pydantic objects → DataFrame (kolom sama dgn training) → Pipeline."""
    df = pd.DataFrame([item.model_dump() for item in items])
    df = df[ml["model"].feature_names_in_]  # urutan kolom = saat training
    preds = ml["model"].predict(df)
    return [int(round(p)) for p in preds]


@app.get("/")
def root():
    return {"message": "Harga Rumah API — buka /docs", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": ml["version"]}


@app.get("/model-info")
def model_info():
    return ml["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(item: HouseFeatures):
    harga = predict_harga([item])[0]
    return PredictionResponse(
        harga_prediksi=harga, model_version=ml["version"]
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    preds = predict_harga(req.items)
    return BatchResponse(
        predictions=preds, n=len(preds), model_version=ml["version"]
    )
