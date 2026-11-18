"""Recompute the (classifier-only) route-continuity column with the fixed metric.

The continuity metric was initially computed with an incorrect adjacency mapping;
this recomputes it for every saved model and patches model_comparison.csv.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from pathsnap.evaluate import point_predictions, route_continuity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
ART = os.path.join(ROOT, "artifacts", "models")


def recompute():
    csv_path = os.path.join(EXP, "results", "model_comparison.csv")
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        name = row["model"].replace(" ", "_")
        path = os.path.join(ART, f"{name}.joblib")
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        conts, npts = 0.0, 0
        for f in sorted(os.listdir(os.path.join(EXP, "student_test"))):
            d = np.load(os.path.join(EXP, "student_test", f))
            proba = model.predict_proba(d["X"])[:, 1]
            pred, _ = point_predictions(d["point_idx"], d["arcs"], proba,
                                        int(d["n_points"]))
            c = route_continuity(pred, d["net_arcs"])
            conts += c * int(d["n_points"])
            npts += int(d["n_points"])
        df.loc[df["model"] == row["model"], "student_continuity"] = conts / npts
        print(f"{row['model']}: continuity {conts/npts:.4f}")

    df.to_csv(csv_path, index=False)
    print("updated", csv_path)


if __name__ == "__main__":
    recompute()
