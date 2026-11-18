"""Compare tuned vs untuned LightGBM on identical metrics (student-test).

Run:  PYTHONPATH=. python3 scripts/compare_tuned.py
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd

from pathsnap.evaluate import evaluate_pairs, point_predictions, route_continuity
from pathsnap.hmm import HMMDecoder, build_network
from pathsnap.io import load_trip
from pathsnap.pipeline import predict_trip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
ART = os.path.join(ROOT, "artifacts", "models")
RES = os.path.join(EXP, "results")


def eval_model(model, hmm=None):
    top1, mf1, ll, cont, miss, npts, npairs, itimes = 0, 0, 0, 0, 0, 0, 0, []
    for f in sorted(os.listdir(os.path.join(EXP, "student_test"))):
        d = np.load(os.path.join(EXP, "student_test", f))
        trip = load_trip(os.path.join(ROOT, "student_test"), f[:-4])
        X, point_idx, arcs, y = d["X"], d["point_idx"], d["arcs"], d["y"]
        true_arc, n = d["true_arc"], int(d["n_points"])
        t0 = time.time()
        proba = model.predict_proba(X)[:, 1]
        itimes.append(time.time() - t0)
        if hmm is None:
            pred, _ = point_predictions(point_idx, arcs, proba, n)
        else:
            res = predict_trip(model, trip, use_hmm=True, hmm=hmm)
            pred = res["final_arcs"]
        top1 += (pred == true_arc).sum()
        present = np.zeros(n, bool)
        np.logical_or.at(present, point_idx, arcs == true_arc[point_idx])
        miss += int((~present).sum())
        cont += route_continuity(pred, d["net_arcs"]) * n
        m = evaluate_pairs(y, proba)
        mf1 += m["macro_f1"] * len(y)
        ll += m["log_loss"] * len(y)
        npts += n; npairs += len(y)
    return dict(top1=top1 / npts, macro_f1=mf1 / npairs, log_loss=ll / npairs,
                continuity=cont / npts, miss_rate=miss / npts,
                inference_time=np.mean(itimes))


def main():
    untuned = joblib.load(os.path.join(ART, "LightGBM.joblib"))
    tuned = joblib.load(os.path.join(ART, "LightGBM_tuned.joblib"))
    hmm = HMMDecoder()

    rows = []
    for label, m in (("untuned", untuned), ("tuned", tuned)):
        r = eval_model(m)
        r_h = eval_model(m, hmm)
        rows.append(dict(model=label, **r, top1_hmm=r_h["top1"],
                         continuity_hmm=r_h["continuity"]))
        print(f"{label}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} "
              f"logloss={r['log_loss']:.4f} | +HMM top1={r_h['top1']:.4f} "
              f"cont={r_h['continuity']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
    print("saved tuned_vs_untuned.csv")


if __name__ == "__main__":
    main()
