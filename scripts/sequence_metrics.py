"""Sequence-level route-reconstruction metrics for the final model.

Computes, for the deduplicated predicted route vs the reference route:
  * exact match rate,
  * LCS-based sequence similarity (how much of the reference route is recovered in order).

Run:  PYTHONPATH=. python3 scripts/sequence_metrics.py
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from pathsnap.hmm import HMMDecoder
from pathsnap.io import load_trip
from pathsnap.pipeline import predict_trip, reconstruct_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
ART = os.path.join(ROOT, "artifacts", "models")
RES = os.path.join(EXP, "results")


def lcs_similarity(a, b):
    """LCS length / len(b)  (fraction of reference route recovered in order)."""
    if len(b) == 0:
        return 1.0
    m, n = len(a), len(b)
    # row of DP, O(m*n)
    prev = np.zeros(n + 1, dtype=int)
    for i in range(1, m + 1):
        cur = np.zeros(n + 1, dtype=int)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return float(prev[n]) / len(b)


def main():
    model = joblib.load(os.path.join(ART, "LightGBM.joblib"))
    hmm = HMMDecoder()
    rows = []
    for f in sorted(os.listdir(os.path.join(EXP, "student_test"))):
        tid = f[:-4]
        trip = load_trip(os.path.join(ROOT, "student_test"), tid)
        ref = reconstruct_route(trip.route)
        res_c = predict_trip(model, trip, use_hmm=False)
        res_h = predict_trip(model, trip, use_hmm=True, hmm=hmm)
        rc = reconstruct_route(res_c["classifier_arcs"])
        rh = reconstruct_route(res_h["final_arcs"])
        rows.append(dict(
            trip=tid,
            lcs_classifier=lcs_similarity(rc, ref),
            lcs_hmm=lcs_similarity(rh, ref),
            exact_classifier=int(np.array_equal(rc, ref)),
            exact_hmm=int(np.array_equal(rh, ref)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "sequence_metrics.csv"), index=False)
    print(df.round(3).to_string(index=False))
    print("\nmean LCS similarity: classifier %.3f -> HMM %.3f"
          % (df.lcs_classifier.mean(), df.lcs_hmm.mean()))
    print("exact route match: classifier %d/%d -> HMM %d/%d"
          % (df.exact_classifier.sum(), len(df), df.exact_hmm.sum(), len(df)))


if __name__ == "__main__":
    main()
