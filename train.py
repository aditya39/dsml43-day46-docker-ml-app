"""
train.py — Latih model "Prediksi Harga Rumah Jabodetabek" lalu simpan artefaknya.

Pemakaian (dari folder project, venv aktif):
  python make_dataset.py         # → data/train.csv (2000 baris)
  python train.py                # → model/model.joblib + model/metadata.json
  python train.py --version v2   # label versi lain (masih diketik tangan —
                                 #   Day 47 urusan versi diserahkan ke MLflow)

Kenapa sklearn Pipeline? Preprocessing (OneHot kota) dan model disimpan sebagai
SATU artefak. API cukup memanggil .predict() pada DataFrame mentah — tidak ada
preprocessing yang ditulis ulang di sisi serving (anti training-serving skew).
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "luas_tanah_m2",
    "luas_bangunan_m2",
    "kamar_tidur",
    "kamar_mandi",
    "jarak_ke_pusat_km",
    "umur_bangunan_tahun",
    "kota",
]
CATEGORICAL = ["kota"]
TARGET = "harga"
RANDOM_STATE = 43


def build_pipeline() -> Pipeline:
    """OneHot(kota) + kolom numerik apa adanya → RandomForest 100 pohon."""
    onehot = OneHotEncoder(handle_unknown="ignore")
    preprocess = ColumnTransformer(
        transformers=[("kota", onehot, CATEGORICAL)],
        remainder="passthrough",  # kolom numerik diteruskan tanpa diubah
    )
    model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Latih model harga rumah")
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--out-dir", default="model")
    ap.add_argument("--version", default="v1", help="label versi, mis. v1")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"❌ {data_path} tidak ada. Jalankan dulu: python make_dataset.py"
        )

    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path, compress=3)  # compress → file lebih kecil

    metadata = {
        "model_name": "harga-rumah-rf",
        "model_version": args.version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {"mae": round(float(mae)), "r2": round(float(r2), 4)},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
        "target": TARGET,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    size_mb = model_path.stat().st_size / 1e6
    print(f"✅ Model tersimpan → {model_path} ({size_mb:.1f} MB)")
    print(f"   versi {args.version} · MAE Rp {mae / 1e6:,.0f} juta · R² {r2:.3f}")
    print(f"   metadata → {meta_path}")


if __name__ == "__main__":
    main()
