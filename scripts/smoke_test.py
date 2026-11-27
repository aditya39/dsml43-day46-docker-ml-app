"""
scripts/smoke_test.py — cek cepat: API hidup, menjawab benar, dan menolak input ngawur.

Pemakaian (dari folder project; cukup Python standar, tanpa install apa pun):
  python scripts/smoke_test.py                              # http://localhost:8000
  python scripts/smoke_test.py --url http://localhost:8080  # port lain
  python scripts/smoke_test.py --require-all                # /model-info & /predict/batch wajib
Atau lewat env: API_URL=http://localhost:8080 (PowerShell: $env:API_URL="http://localhost:8080")

Exit code 0 = semua lolos, 1 = ada yang gagal (bisa dipakai di CI/CD).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

CONTOH = {
    "luas_tanah_m2": 120,
    "luas_bangunan_m2": 100,
    "kamar_tidur": 3,
    "kamar_mandi": 2,
    "jarak_ke_pusat_km": 25.0,
    "umur_bangunan_tahun": 5,
    "kota": "Depok",
}


def call(method: str, url: str, payload: dict | None = None, timeout: float = 10):
    """Kirim request HTTP, kembalikan (status_code, body_json_atau_teks)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:  # 4xx / 5xx tetap kita baca body-nya
        raw, status = e.read(), e.code
    try:
        return status, json.loads(raw or b"null")
    except json.JSONDecodeError:
        return status, raw.decode("utf-8", errors="replace")


def wait_until_up(base: str, seconds: int) -> bool:
    """Container butuh beberapa detik untuk load model — tunggu /health."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            status, _ = call("GET", f"{base}/health", timeout=3)
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test Harga Rumah API")
    ap.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--wait", type=int, default=30, help="detik menunggu API siap")
    ap.add_argument("--require-all", action="store_true",
                    help="gagal kalau /model-info atau /predict/batch belum ada")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"Smoke test -> {base}")
    if not wait_until_up(base, args.wait):
        print(f"[FAIL] API tidak menjawab di {base}/health dalam {args.wait} detik.")
        print("       Cek: container jalan? (docker ps)  port benar?  docker logs <nama>")
        return 1

    results: list[tuple[str, str, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(("PASS" if ok else "FAIL", name, detail))

    # 1) health
    status, body = call("GET", f"{base}/health")
    check("GET /health -> 200", status == 200 and isinstance(body, dict)
          and body.get("status") == "ok", f"{status} {body}")

    # 2) prediksi valid
    status, body = call("POST", f"{base}/predict", CONTOH)
    harga = body.get("harga_prediksi") if isinstance(body, dict) else None
    ok = status == 200 and isinstance(harga, int) and 100_000_000 < harga < 50_000_000_000
    ok = ok and bool(body.get("model_version"))
    detail = f"{status} harga=Rp {harga:,}" if isinstance(harga, int) else f"{status} {body}"
    check("POST /predict (valid) -> 200 + harga wajar + model_version", ok,
          detail + (f" versi={body.get('model_version')}" if isinstance(body, dict) else ""))

    # 3) validasi: tipe salah, angka negatif, kota tak dikenal, field hilang -> 422
    salah = [
        ("teks di kolom angka", {**CONTOH, "luas_tanah_m2": "seratus"}),
        ("luas_tanah_m2 negatif", {**CONTOH, "luas_tanah_m2": -50}),
        ("kota tidak dikenal", {**CONTOH, "kota": "Surabaya"}),
        ("field kamar_tidur hilang", {k: v for k, v in CONTOH.items() if k != "kamar_tidur"}),
    ]
    for label, payload in salah:
        status, body = call("POST", f"{base}/predict", payload)
        hint = "" if status == 422 else "  <- cek Field(...) / Literal di app/schemas.py"
        check(f"POST /predict ({label}) -> 422", status == 422, f"{status}{hint}")

    # 4) endpoint tambahan (Exercise) — opsional kecuali --require-all
    status, body = call("GET", f"{base}/model-info")
    if status == 404 and not args.require_all:
        results.append(("SKIP", "GET /model-info", "404 — belum dibuat (Exercise)"))
    else:
        ok = status == 200 and isinstance(body, dict) and "model_version" in body
        check("GET /model-info -> 200 + model_version", ok, f"{status}")

    batch = {"items": [CONTOH, {**CONTOH, "kota": "Jakarta Selatan"}]}
    status, body = call("POST", f"{base}/predict/batch", batch)
    if status == 404 and not args.require_all:
        results.append(("SKIP", "POST /predict/batch", "404 — belum dibuat (Exercise bonus)"))
    else:
        preds = body.get("predictions") if isinstance(body, dict) else None
        ok = status == 200 and isinstance(preds, list) and len(preds) == 2
        check("POST /predict/batch (2 rumah) -> 200 + 2 prediksi", ok, f"{status} {preds}")

    for tag, name, detail in results:
        print(f"[{tag}] {name}  {detail}")
    failed = sum(1 for tag, _, _ in results if tag == "FAIL")
    passed = sum(1 for tag, _, _ in results if tag == "PASS")
    print(f"\n{passed} lolos, {failed} gagal, {len(results) - passed - failed} dilewati")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
