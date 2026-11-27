# Dockerfile — versi OPTIMIZED (multi-stage, slim, non-root, healthcheck).
# Build : docker build -t harga-rumah-api:1.0 .
# Run   : docker run -d --name harga-api -p 8000:8000 harga-rumah-api:1.0
# Syarat: model/model.joblib ada (sudah ikut di repo; buat ulang: python train.py).

# ---------- Stage 1: builder — install dependency ke virtualenv ----------
FROM python:3.11-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
# install → buang pip/setuptools (tidak perlu saat serving) → buang folder
# tests/ bawaan numpy/scipy/pandas/sklearn (±115 MB). Dibuang DI SINI, di
# stage builder, supaya stage runtime cuma menerima venv yang sudah bersih.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools \
    && find /opt/venv -type d -name tests -prune -exec rm -rf {} +

# ---------- Stage 2: runtime — hanya yang perlu untuk serving ----------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/model/model.joblib
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY model/ ./model/
COPY app/ ./app/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
