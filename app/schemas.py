"""
app/schemas.py — "satpam di pintu": skema request & response.

Pydantic memvalidasi data SEBELUM model disentuh. Request yang tidak valid
(tipe salah, angka negatif, kota tidak dikenal, field hilang) otomatis ditolak
FastAPI dengan HTTP 422 — tanpa satu baris pun kode validasi manual.

Catatan desain: batas di sini adalah aturan "masuk akal" (luas > 0, kamar ≥ 1),
BUKAN rentang data training. Input yang valid tapi jauh dari data training
(mis. rumah 70 km dari pusat kota) itu urusan monitoring drift — Day 47.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Kota = Literal[
    "Jakarta Selatan",
    "Jakarta Barat",
    "Jakarta Timur",
    "Depok",
    "Tangerang Selatan",
    "Bekasi",
    "Bogor",
]


class HouseFeatures(BaseModel):
    """Satu rumah = satu baris fitur. Nama field = nama kolom training."""

    luas_tanah_m2: int = Field(gt=0, le=10_000, examples=[120])
    luas_bangunan_m2: int = Field(gt=0, le=10_000, examples=[100])
    kamar_tidur: int = Field(ge=1, le=20, examples=[3])
    kamar_mandi: int = Field(ge=1, le=20, examples=[2])
    jarak_ke_pusat_km: float = Field(ge=0, le=200, examples=[25.0])
    umur_bangunan_tahun: int = Field(ge=0, le=200, examples=[5])
    kota: Kota = Field(examples=["Depok"])


class PredictionResponse(BaseModel):
    harga_prediksi: int = Field(description="Prediksi harga (Rupiah)")
    model_version: str = Field(description="Versi model yang menjawab")


class BatchRequest(BaseModel):
    items: list[HouseFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[int]
    n: int
    model_version: str
