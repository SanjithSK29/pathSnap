"""Train and compare all models with group-aware cross-validation and
student-test evaluation.

Run:  PYTHONPATH=. python3 scripts/train_models.py
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.evaluate import evaluate_pairs, point_predictions, route_continuity
from pathsnap.models import MODELS, compute_sample_weights, fit_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
ART = os.path.join(ROOT, "artifacts", "models")
os.makedirs(ART, exist_ok=True)


def load_train():
    d = np.load(os.path.join(EXP, "train_pairs.npz"))
    return d["X"], d["y"], d["trip_idx"]


def load_student_test():
    outdir = os.path.join(EXP, "student_test")
    trips = sorted(f[:-4] for f in os.listdir(outdir) if f.endswith(".npz"))
    data = {}
    for t in trips:
        data[t] = np.load(os.path.join(outdir, f"{t}.npz"))
    return data


def run_cv(X, y, groups, spec, n_splits=5):
    kf = GroupKFold(n_splits=n_splits)
    macro_f1s, loglosses, precs, recs = [], [], [], []
    for tr, va in kf.split(X, y, groups=groups):
        sw = compute_sample_weights(y[tr])
        est, _ = fit_model(spec, X[tr], y[tr], sample_weight=sw)
        proba = est.predict_proba(X[va])[:, 1]
        m = evaluate_pairs(y[va], proba)
        macro_f1s.append(m["macro_f1"])
        loglosses.append(m["log_loss"])
        precs.append(m["precision"])
        recs.append(m["recall"])
    return dict(
        macro_f1=float(np.mean(macro_f1s)), macro_f1_std=float(np.std(macro_f1s)),
        log_loss=float(np.mean(loglosses)),
        precision=float(np.mean(precs)), recall=float(np.mean(recs)),
    )


def evaluate_student_test(model, student_data):
    top1s, macro_f1s, loglosses, conts, misses = [], [], [], [], []
    n_points_total = 0
    n_pairs_total = 0
    inf_times = []
    for t, d in student_data.items():
        X = d["X"]; point_idx = d["point_idx"]; arcs = d["arcs"]; y = d["y"]
        true_arc = d["true_arc"]; n_points = int(d["n_points"])
        t0 = time.time()
        proba = model.predict_proba(X)[:, 1]
        inf_times.append(time.time() - t0)
        # point-level top-1
        pred, _ = point_predictions(point_idx, arcs, proba, n_points)
        top1s.append((pred == true_arc).sum())
        # candidate miss rate: true arc absent from the candidate set
        present = np.zeros(n_points, dtype=bool)
        np.logical_or.at(present, point_idx, arcs == true_arc[point_idx])
        misses.append(int((~present).sum()))
        # pair-level
        m = evaluate_pairs(y, proba)
        macro_f1s.append(m["macro_f1"] * len(y))
        loglosses.append(m["log_loss"] * len(y))
        # route continuity (classifier-only)
        conts.append(route_continuity(pred, d["net_arcs"]) * n_points)
        n_points_total += n_points
        n_pairs_total += len(y)
    return dict(
        top1=float(np.sum(top1s) / n_points_total),
        macro_f1=float(np.sum(macro_f1s) / n_pairs_total),
        log_loss=float(np.sum(loglosses) / n_pairs_total),
        continuity=float(np.sum(conts) / n_points_total),
        miss_rate=float(np.sum(misses) / n_points_total),
        inference_time=float(np.mean(inf_times)),
    )


def main():
    X, y, groups = load_train()
    student_data = load_student_test()
    print(f"train pairs {X.shape}, student_test trips {len(student_data)}")

    # resume: load any previously completed rows
    csv_path = os.path.join(EXP, "results", "model_comparison.csv")
    rows = []
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = set(prev["model"].tolist())

    for spec in MODELS:
        name = spec["name"]
        if name in done:
            print(f"\n=== {name} ({spec['family']}) — already done, skipping ===", flush=True)
            continue
        print(f"\n=== {name} ({spec['family']}) ===", flush=True)
        t0 = time.time()
        cv = run_cv(X, y, groups, spec)
        print(f"  CV: macroF1={cv['macro_f1']:.4f}+-{cv['macro_f1_std']:.4f} "
              f"logloss={cv['log_loss']:.4f} ({time.time()-t0:.0f}s)", flush=True)

        sw = compute_sample_weights(y)
        est, fit_time = fit_model(spec, X, y, sample_weight=sw)
        joblib.dump(est, os.path.join(ART, f"{name.replace(' ', '_')}.joblib"))

        st = evaluate_student_test(est, student_data)
        print(f"  student_test: top1={st['top1']:.4f} macroF1={st['macro_f1']:.4f} "
              f"logloss={st['log_loss']:.4f} continuity={st['continuity']:.4f} "
              f"infer={st['inference_time']:.3f}s", flush=True)

        rows.append(dict(
            model=name, family=spec["family"], key_config=spec["key_config"],
            cv_macro_f1=cv["macro_f1"], cv_macro_f1_std=cv["macro_f1_std"],
            cv_log_loss=cv["log_loss"],
            cv_precision=cv["precision"], cv_recall=cv["recall"],
            student_top1=st["top1"], student_macro_f1=st["macro_f1"],
            student_log_loss=st["log_loss"], student_continuity=st["continuity"],
            inference_time=st["inference_time"], fit_time=fit_time,
        ))

        # persist incrementally so partial runs survive interruption
        pd.DataFrame(rows).to_csv(os.path.join(EXP, "results", "model_comparison.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP, "results", "model_comparison.csv"), index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
