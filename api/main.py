"""PathSnap FastAPI backend.

Laptop 1 hosts the trained classifier, the HMM/Viterbi decoder and the road-network
information.  The UI (Laptop 2) sends a trip id; the API runs the full prediction
pipeline, optionally computes accuracy against its *server-side* reference route,
and returns the predicted route + results.

Run:
    PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.hmm import HMMDecoder
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import predict_trip, reconstruct_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load config.json if present (env vars take precedence)
CONFIG = {}
_cfg_path = os.path.join(ROOT, "config.json")
if os.path.exists(_cfg_path):
    try:
        import json as _json
        with open(_cfg_path) as _f:
            CONFIG = _json.load(_f)
    except Exception:
        CONFIG = {}

_hmm_cfg = CONFIG.get("hmm", {})
DATA_DIR = os.environ.get("PATHSNAP_DATA_DIR",
                          os.path.join(ROOT, CONFIG.get("deployment", {}).get("data_dir", "student_test")))
MODEL_PATH = os.environ.get("PATHSNAP_MODEL",
                            os.path.join(ROOT, CONFIG.get("deployment", {}).get("model",
                                "artifacts/models/LightGBM.joblib")))
HMM_BETA = float(os.environ.get("PATHSNAP_HMM_BETA", _hmm_cfg.get("beta", 20.0)))
HMM_DISCONNECTED = float(os.environ.get("PATHSNAP_HMM_DISCONNECTED",
                                       _hmm_cfg.get("disconnected_penalty", 1e-6)))
HMM_SWITCH = float(os.environ.get("PATHSNAP_HMM_SWITCH", _hmm_cfg.get("switch_penalty", 1.0)))
EXPOSE_REFERENCE = os.environ.get("PATHSNAP_EXPOSE_REFERENCE", "0") == "1"

app = FastAPI(title="PathSnap", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

model = None
decoder = HMMDecoder(beta=HMM_BETA, disconnected_penalty=HMM_DISCONNECTED,
                     switch_penalty=HMM_SWITCH)


@app.on_event("startup")
def load_artifacts():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"[PathSnap] loaded model {MODEL_PATH}")
    print(f"[PathSnap] data dir {DATA_DIR}")


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/trips")
def trips():
    return {"trips": list_trips(DATA_DIR)}


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id
    path = os.path.join(DATA_DIR, tid)
    if not os.path.isdir(path):
        raise HTTPException(404, f"unknown trip {tid}")
    trip = load_trip(DATA_DIR, tid)

    res = predict_trip(model, trip, use_hmm=True, hmm=decoder)
    route = reconstruct_route(res["final_arcs"])

    # server-side evaluation if the reference route is available.
    # The reference route (ground truth) is NEVER returned to the UI.
    accuracy = None
    if trip.route is not None:
        true_arc, _ = _reference_arcs(trip)
        accuracy = float((res["final_arcs"] == true_arc).mean())

    resp = {
        "trip_id": tid,
        "n_points": trip.n_points,
        "predicted_route": route.astype(int).tolist(),
        "classifier_route": reconstruct_route(res["classifier_arcs"]).astype(int).tolist(),
        "accuracy": accuracy,
        "track": trip.track.astype(float).tolist(),
        "nodes": trip.nodes.astype(float).tolist(),
        "arcs": trip.arcs.astype(int).tolist(),
        "predicted_arcs": res["final_arcs"].astype(int).tolist(),
    }
    # development-only: optionally expose the reference route (off by default)
    if EXPOSE_REFERENCE and trip.route is not None:
        resp["reference_route"] = trip.route.astype(int).tolist()
    return resp


def _reference_arcs(trip):
    from pathsnap.labeling import label_trip
    return label_trip(trip)
