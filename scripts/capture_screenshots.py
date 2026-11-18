"""Capture UI and API screenshots for the report.

Starts the FastAPI backend (if a model is available), screenshots the UI with
headless Chrome, and captures a successful API response transcript.

Run:  PYTHONPATH=. python3 scripts/capture_screenshots.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
ART = os.path.join(ROOT, "artifacts", "models")
os.makedirs(FIG, exist_ok=True)


def find_model():
    for cand in ("LightGBM_tuned.joblib", "LightGBM.joblib", "XGBoost.joblib",
                 "Logistic_Regression.joblib"):
        p = os.path.join(ART, cand)
        if os.path.exists(p):
            return p
    return None


def main():
    model = find_model()
    if model is None:
        print("no model found; run training first")
        return

    env = dict(os.environ)
    env["PATHSNAP_MODEL"] = model
    env["PATHSNAP_DATA_DIR"] = os.path.join(ROOT, "student_test")
    env["PYTHONPATH"] = ROOT
    proc = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8011"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(30):
            try:
                r = requests.get("http://127.0.0.1:8011/health", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            print("API did not start")
            return

        # API response transcript
        trips = requests.get("http://127.0.0.1:8011/trips").json()["trips"]
        tid = trips[0]
        resp = requests.post("http://127.0.0.1:8011/predict",
                             json={"trip_id": tid}, timeout=120).json()
        transcript = [
            "$ curl -s http://127.0.0.1:8011/trips",
            json.dumps({"trips": trips}, indent=2)[:600],
            f"\n$ curl -s -X POST http://127.0.0.1:8011/predict -d '{{\"trip_id\":\"{tid}\"}}'",
            json.dumps({k: (v if k not in ("track", "nodes", "arcs",
                         "reference_route", "predicted_arcs") else f"<{len(v)} values>")
                        for k, v in resp.items()}, indent=2),
        ]
        with open(os.path.join(FIG, "api_response.txt"), "w") as f:
            f.write("\n".join(transcript))
        print("captured API transcript")

        # UI screenshot (serve ui dir and screenshot the page via a tiny proxy)
        ui_proc = subprocess.Popen(
            ["python3", "-m", "http.server", "8012", "--directory",
             os.path.join(ROOT, "ui")],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        subprocess.run([
            "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
            "--hide-scrollbars", "--window-size=1400,950",
            f"--screenshot={os.path.join(FIG, 'ui_screenshot.png')}",
            "http://127.0.0.1:8012/index.html",
        ], check=True, capture_output=True, timeout=120)
        ui_proc.terminate()
        print("captured UI screenshot")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
