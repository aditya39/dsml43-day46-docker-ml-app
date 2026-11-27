"""
ui/app.py — Streamlit UI untuk model "Prediksi Harga Rumah Jabodetabek".

Satu file, dua mode. Mode dipilih otomatis dari environment variable API_URL:
  API_URL di-set  → mode API   : isi form dikirim ke FastAPI (POST {API_URL}/predict)
  API_URL kosong  → mode lokal : model/model.joblib dimuat langsung oleh Streamlit

Jalankan dari folder project (venv aktif):
  streamlit run ui/app.py                                           # mode lokal
  API_URL=http://localhost:8000 streamlit run ui/app.py             # mode API (macOS/Linux)
  $env:API_URL="http://localhost:8000"; streamlit run ui/app.py     # mode API (PowerShell)
Lalu buka http://localhost:8501

Di docker compose: API_URL=http://api:8000 — "api" adalah NAMA SERVICE di
docker-compose.yml. Di dalam container, "localhost" berarti container itu sendiri.

Variabel opsional: MODEL_PATH, METADATA_PATH, DATA_PATH (default: folder project).
"""
from __future__ import annotations

import json
import os
import platform
import socket
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import sklearn
import streamlit as st

# ----------------------------- konfigurasi -----------------------------
ROOT = Path(__file__).resolve().parent.parent  # folder project (di image: /app)
API_URL = os.getenv("API_URL", "").strip().rstrip("/")
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model" / "model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", MODEL_PATH.with_name("metadata.json")))
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "train.csv"))
TIMEOUT_DETIK = 10

# Harus sama persis dengan Literal "Kota" di app/schemas.py
KOTA = [
    "Jakarta Selatan",
    "Jakarta Barat",
    "Jakarta Timur",
    "Depok",
    "Tangerang Selatan",
    "Bekasi",
    "Bogor",
]


class PrediksiGagal(Exception):
    """Error yang pesannya aman & jelas untuk ditampilkan ke pengguna."""


# ----------------------------- format angka -----------------------------
def rupiah(x: float) -> str:
    """1170610000 → 'Rp 1.170.610.000' (titik = pemisah ribuan)."""
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def ringkas(x: float) -> str:
    """1170610000 → 'Rp 1,17 miliar' · 850000000 → 'Rp 850 juta'."""
    if abs(x) >= 1_000_000_000:
        return "Rp " + f"{x / 1e9:.2f}".replace(".", ",") + " miliar"
    return "Rp " + f"{x / 1e6:,.0f}".replace(",", ".") + " juta"


# ----------------------------- mode lokal -----------------------------
@st.cache_resource(show_spinner="Memuat model ...")
def load_model_lokal(path: str):
    """Load SEKALI per proses Streamlit, lalu dipakai ulang (sama seperti lifespan di API)."""
    return joblib.load(path)


def baca_metadata_lokal() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def prediksi_lokal(rumah: dict) -> dict:
    if not MODEL_PATH.exists():
        raise PrediksiGagal(
            f"File model tidak ditemukan: `{MODEL_PATH}`. "
            "Jalankan dulu `python train.py` dari folder project."
        )
    model = load_model_lokal(str(MODEL_PATH))
    # Sama persis dengan predict_harga() di app/main.py: urutan kolom = saat training
    df = pd.DataFrame([rumah])[model.feature_names_in_]
    harga = int(round(model.predict(df)[0]))
    versi = baca_metadata_lokal().get("model_version", "unknown")
    return {"harga_prediksi": harga, "model_version": versi}


# ----------------------------- mode API -----------------------------
def petunjuk_api() -> str:
    return (
        "Cek:\n"
        "- **Tanpa Docker:** apakah `uvicorn app.main:app --port 8000` sedang jalan "
        "di terminal lain? Apakah `API_URL` menunjuk ke port yang sama?\n"
        "- **Dengan Docker:** jalankan `docker compose ps` — service `api` harus `healthy`. "
        "Lihat log dengan `docker compose logs api`."
    )


def prediksi_api(rumah: dict) -> dict:
    url = f"{API_URL}/predict"
    try:
        resp = requests.post(url, json=rumah, timeout=TIMEOUT_DETIK)
    except requests.exceptions.ConnectionError:
        raise PrediksiGagal(
            f"Tidak bisa terhubung ke API di `{API_URL}`.\n\n" + petunjuk_api()
        ) from None
    except requests.exceptions.Timeout:
        raise PrediksiGagal(
            f"API di `{API_URL}` tidak menjawab dalam {TIMEOUT_DETIK} detik."
        ) from None

    if resp.status_code == 422:  # ditolak Pydantic (app/schemas.py)
        detail = resp.json().get("detail", [])
        pesan = "\n".join(
            f"- `{'.'.join(str(p) for p in d.get('loc', [])[1:])}`: {d.get('msg')}"
            for d in detail
        )
        raise PrediksiGagal(f"API menolak input (HTTP 422):\n{pesan}")
    if resp.status_code != 200:
        raise PrediksiGagal(f"API menjawab HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


@st.cache_data(ttl=5, show_spinner=False)
def _get_json(url: str) -> dict:
    """GET → JSON. Hanya jawaban SUKSES yang di-cache (error tidak di-cache)."""
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    return resp.json()


def cek_health(api_url: str) -> dict | None:
    """GET /health — None kalau API tidak bisa dihubungi."""
    try:
        return _get_json(f"{api_url}/health")
    except requests.RequestException:
        return None


def ambil_metadata_api(api_url: str) -> dict | None:
    """GET /model-info — isi model/metadata.json milik container API."""
    try:
        return _get_json(f"{api_url}/model-info")
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["harga_miliar"] = df["harga"] / 1e9
    return df


# =============================== HALAMAN ===============================
st.set_page_config(
    page_title="Harga Rumah Jabodetabek",
    page_icon="🏠",
    layout="wide",
)

# --- Sidebar: "aplikasi ini sedang berjalan di mana?" ---
with st.sidebar:
    st.header("⚙️ Lingkungan")
    if API_URL:
        st.markdown(f"**Mode:** 🌐 API  \n`{API_URL}`")
        health = cek_health(API_URL)
        if health:
            st.success(f"API sehat · model `{health.get('model_version')}`")
        else:
            st.error("API tidak bisa dihubungi")
    else:
        st.markdown("**Mode:** 💻 lokal  \nmodel dimuat langsung dari `model/model.joblib`")

    st.markdown(
        f"| Info | Nilai |\n|---|---|\n"
        f"| Hostname | `{socket.gethostname()}` |\n"
        f"| Python | `{platform.python_version()}` |\n"
        f"| OS | `{platform.system()} {platform.machine()}` |\n"
        f"| scikit-learn | `{sklearn.__version__}` |"
    )
    st.caption(
        "Di laptop, hostname = nama laptop kalian. Di dalam Docker, hostname = ID container "
        "dan Python-nya milik image, bukan milik laptop."
    )

# Bagian C · Eksperimen 1: ubah teks judul di bawah, lalu rebuild image UI.
st.title("🏠 Prediksi Harga Rumah Jabodetabek")
st.caption(
    "Day 46 · DSML Batch 43 — RandomForest pada data sintetis. "
    "Angka hanya untuk latihan, bukan harga pasar sungguhan."
)

tab_prediksi, tab_data, tab_model = st.tabs(["🔮 Prediksi", "📊 Data", "ℹ️ Model"])

# ------------------------------ PREDIKSI ------------------------------
with tab_prediksi:
    col_form, col_hasil = st.columns([3, 2], gap="large")

    with col_form:
        # Batas tiap input = batas di app/schemas.py, jadi API tidak akan menolak (422)
        with st.form("form_rumah"):
            kota = st.selectbox("Kota", KOTA, index=KOTA.index("Depok"))
            c1, c2 = st.columns(2)
            luas_tanah = c1.number_input(
                "Luas tanah (m²)", min_value=1, max_value=10_000, value=120, step=10
            )
            luas_bangunan = c2.number_input(
                "Luas bangunan (m²)", min_value=1, max_value=10_000, value=100, step=10
            )
            kamar_tidur = c1.slider("Kamar tidur", min_value=1, max_value=10, value=3)
            kamar_mandi = c2.slider("Kamar mandi", min_value=1, max_value=10, value=2)
            jarak = c1.number_input(
                "Jarak ke pusat kota (km)",
                min_value=0.0, max_value=200.0, value=25.0, step=0.5,
            )
            umur = c2.number_input(
                "Umur bangunan (tahun)", min_value=0, max_value=200, value=5, step=1
            )
            submitted = st.form_submit_button(
                "🔮 Prediksi harga", type="primary", width="stretch"
            )

    if submitted:
        # Nama key = nama field di HouseFeatures (app/schemas.py) = nama kolom training
        rumah = {
            "luas_tanah_m2": int(luas_tanah),
            "luas_bangunan_m2": int(luas_bangunan),
            "kamar_tidur": int(kamar_tidur),
            "kamar_mandi": int(kamar_mandi),
            "jarak_ke_pusat_km": float(jarak),
            "umur_bangunan_tahun": int(umur),
            "kota": kota,
        }
        try:
            with st.spinner("Menghitung ..."):
                respon = prediksi_api(rumah) if API_URL else prediksi_lokal(rumah)
            st.session_state["hasil"] = {"rumah": rumah, "respon": respon, "error": None}
        except PrediksiGagal as e:
            st.session_state["hasil"] = {"rumah": rumah, "respon": None, "error": str(e)}

    with col_hasil:
        st.subheader("Hasil")
        hasil = st.session_state.get("hasil")
        if hasil is None:
            st.info("Isi data rumah di kiri, lalu klik **Prediksi harga**.")
        elif hasil["error"]:
            st.error(hasil["error"])
        else:
            harga = hasil["respon"]["harga_prediksi"]
            rumah = hasil["rumah"]
            st.metric("Perkiraan harga", ringkas(harga))
            st.markdown(f"### {rupiah(harga)}")
            per_m2 = harga / rumah["luas_bangunan_m2"]
            st.caption(
                f"{rumah['kota']} · LT {rumah['luas_tanah_m2']} m² · "
                f"LB {rumah['luas_bangunan_m2']} m² · ≈ {ringkas(per_m2)} per m² bangunan"
            )
            sumber = f"API `{API_URL}`" if API_URL else "model lokal (`model/model.joblib`)"
            st.caption(
                f"Model `{hasil['respon']['model_version']}` · dijawab oleh {sumber}"
            )
            with st.expander("Lihat JSON request & response"):
                st.markdown("Request (`POST /predict`):")
                st.code(json.dumps(rumah, indent=2), language="json")
                st.markdown("Response:")
                st.code(json.dumps(hasil["respon"], indent=2), language="json")

# -------------------------------- DATA --------------------------------
with tab_data:
    if not DATA_PATH.exists():
        st.info(f"`{DATA_PATH}` tidak ada. Buat dengan `python make_dataset.py`.")
    else:
        df = load_data(str(DATA_PATH))
        pilih = st.multiselect("Filter kota", KOTA, default=KOTA)
        dff = df[df["kota"].isin(pilih)]
        if dff.empty:
            st.warning("Pilih minimal satu kota.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Jumlah rumah", f"{len(dff):,}".replace(",", "."))
            m2.metric("Median harga", ringkas(dff["harga"].median()))
            m3.metric("Median luas tanah", f"{dff['luas_tanah_m2'].median():.0f} m²")

            urutan = (
                dff.groupby("kota")["harga"].median().sort_values(ascending=False).index.tolist()
            )
            g1, g2 = st.columns(2)
            with g1:
                fig = px.box(
                    dff, x="kota", y="harga_miliar", color="kota",
                    category_orders={"kota": urutan},
                    labels={"kota": "Kota", "harga_miliar": "Harga (Rp miliar)"},
                    title="Sebaran harga per kota",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig)
            with g2:
                fig = px.scatter(
                    dff, x="luas_tanah_m2", y="harga_miliar", color="kota",
                    category_orders={"kota": urutan}, opacity=0.6,
                    labels={
                        "luas_tanah_m2": "Luas tanah (m²)",
                        "harga_miliar": "Harga (Rp miliar)",
                        "kota": "Kota",
                    },
                    title="Luas tanah vs harga",
                )
                st.plotly_chart(fig)

            with st.expander("Lihat data mentah"):
                st.dataframe(dff.drop(columns="harga_miliar"), hide_index=True)
                st.caption(f"Sumber: `data/train.csv` · {len(dff)} baris")

# -------------------------------- MODEL --------------------------------
with tab_model:
    if API_URL:
        meta = ambil_metadata_api(API_URL)
        asal = f"`GET {API_URL}/model-info`"
    else:
        meta = baca_metadata_lokal()
        asal = "`model/metadata.json`"

    if not meta:
        st.warning(f"Info model belum tersedia (sumber: {asal}).")
    else:
        st.caption(f"Sumber: {asal}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Versi model", meta.get("model_version", "?"))
        mae = meta.get("metrics", {}).get("mae")
        k2.metric("MAE", ringkas(mae) if mae is not None else "?",
                  help="Rata-rata selisih prediksi vs harga asli pada data uji")
        k3.metric("R²", meta.get("metrics", {}).get("r2", "?"))
        k4.metric("Data latih", f"{meta.get('n_train', '?')} rumah")

        st.markdown(
            f"Dilatih **{meta.get('trained_at', '?')}** dengan Python "
            f"`{meta.get('python_version', '?')}` dan scikit-learn "
            f"`{meta.get('sklearn_version', '?')}`."
        )
        if not API_URL and meta.get("sklearn_version") not in (None, sklearn.__version__):
            st.warning(
                f"scikit-learn di environment ini `{sklearn.__version__}`, sedangkan model "
                f"dilatih dengan `{meta.get('sklearn_version')}`. File .joblib bisa gagal "
                "dimuat atau hasilnya berbeda. Inilah masalah *works on my machine* "
                "yang diselesaikan Docker."
            )
        with st.expander("Metadata lengkap (JSON)"):
            st.json(meta)
