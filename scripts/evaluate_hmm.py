"""Compare classifier-only vs classifier+HMM/Viterbi on the student-test set.

Run:  PYTHONPATH=. python3 scripts/evaluate_hmm.py --model <name>
"""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from pathsnap.evaluate import route_continuity
from pathsnap.hmm import HMMDecoder, build_network
from pathsnap.io import load_trip
from pathsnap.pipeline import predict_trip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
ART = os.path.join(ROOT, "artifacts", "models")
RES = os.path.join(EXP, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model name e.g. 'LightGBM'")
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--disconnected", type=float, default=1e-6)
    args = ap.parse_args()

    # model name maps directly to artifacts/models/<name>.joblib
    model_name = args.model or "LightGBM"
    model_path = os.path.join(ART, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(ART, "LightGBM.joblib")
    model = joblib.load(model_path)
    print(f"model: {model_path}")

    decoder = HMMDecoder(beta=args.beta, disconnected_penalty=args.disconnected)

    rows = []
    top1_clf, top1_hmm = [], []
    cont_clf, cont_hmm = [], []
    for f in sorted(os.listdir(os.path.join(EXP, "student_test"))):
        tid = f[:-4]
        trip = load_trip(os.path.join(ROOT, "student_test"), tid)
        res = predict_trip(model, trip, use_hmm=False)
        res_h = predict_trip(model, trip, use_hmm=True, hmm=decoder)

        true_arc = np.load(os.path.join(EXP, "student_test", f))["true_arc"]

        t1c = float((res["classifier_arcs"] == true_arc).mean())
        t1h = float((res_h["final_arcs"] == true_arc).mean())
        cc = route_continuity(res["classifier_arcs"], trip.arcs)
        ch = route_continuity(res_h["final_arcs"], trip.arcs)
        top1_clf.append(t1c); top1_hmm.append(t1h)
        cont_clf.append(cc); cont_hmm.append(ch)
        rows.append(dict(trip=tid, top1_classifier=t1c, top1_hmm=t1h,
                         continuity_classifier=cc, continuity_hmm=ch))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "hmm_comparison.csv"), index=False)
    print(df.round(4).to_string(index=False))
    print("\n=== aggregate (unweighted mean over trips) ===")
    print(f"Top-1: classifier {np.mean(top1_clf):.4f} -> HMM {np.mean(top1_hmm):.4f}")
    print(f"Continuity: classifier {np.mean(cont_clf):.4f} -> HMM {np.mean(cont_hmm):.4f}")


if __name__ == "__main__":
    main()
