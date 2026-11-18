"""Bayesian hyperparameter optimisation of the selected model.

Uses Optuna's TPE sampler (a Bayesian, tree-structured Parzen estimator) with a
group-aware cross-validation objective (groups = trips).

Two objectives are available:
  * ``top1``     — observation-level Top-1 accuracy (the primary metric),
  * ``macro_f1`` — pair-level Macro F1.

Run:  PYTHONPATH=. python3 scripts/bayes_opt.py --model LightGBM --trials 20 --objective top1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.evaluate import evaluate_pairs
from pathsnap.models import compute_sample_weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
RES = os.path.join(EXP, "results")
ART = os.path.join(ROOT, "artifacts", "models")
os.makedirs(RES, exist_ok=True)
os.makedirs(ART, exist_ok=True)

N_FOLDS = 3


def _argmax_per_point(point_idx, arcs, scores, n_points):
    order = np.lexsort((-scores, point_idx))
    idx = point_idx[order]
    a = arcs[order]
    first = np.empty(len(idx), dtype=bool)
    first[0] = True
    first[1:] = idx[1:] != idx[:-1]
    pred = np.full(n_points, -1, dtype=np.int64)
    pred[idx[first]] = a[first]
    return pred


def _top1_objective(trial, X, y, groups, point_idx, arcs, true_arc, make_est):
    params = trial.params
    kf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for tr, va in kf.split(X, y, groups=groups):
        est = make_est(params)
        est.fit(X[tr], y[tr], sample_weight=compute_sample_weights(y[tr]))
        proba = est.predict_proba(X[va])[:, 1]
        # remap point indices of the validation fold to 0..n_val-1
        p = point_idx[va]
        uniq, inv = np.unique(p, return_inverse=True)
        pred = _argmax_per_point(inv, np.asarray(arcs[va], dtype=np.int64),
                                 proba, len(uniq))
        scores.append(float((pred == true_arc[uniq]).mean()))
    return float(np.mean(scores))


def _macro_f1_objective(trial, X, y, groups, point_idx, true_arc, make_est):
    params = trial.params
    kf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for tr, va in kf.split(X, y, groups=groups):
        est = make_est(params)
        est.fit(X[tr], y[tr], sample_weight=compute_sample_weights(y[tr]))
        proba = est.predict_proba(X[va])[:, 1]
        scores.append(evaluate_pairs(y[va], proba)["macro_f1"])
    return float(np.mean(scores))


def _make_lgbm(params):
    from lightgbm import LGBMClassifier
    p = dict(params)
    p["n_jobs"] = -1
    p["random_state"] = 0
    p["verbose"] = -1
    return LGBMClassifier(**p)


def _lgbm_space(trial):
    trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    trial.suggest_int("num_leaves", 15, 255)
    trial.suggest_int("min_child_samples", 5, 100)
    trial.suggest_float("subsample", 0.5, 1.0)
    trial.suggest_float("colsample_bytree", 0.5, 1.0)
    trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
    trial.suggest_int("n_estimators", 150, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LightGBM", choices=["LightGBM"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--objective", default="top1", choices=["top1", "macro_f1"])
    args = ap.parse_args()

    d = np.load(os.path.join(EXP, "train_pairs.npz"))
    X, y, groups = d["X"], d["y"], d["trip_idx"]
    point_idx = d["point_idx"]
    arcs = d["arcs"]

    # global point id and true arc per point (for the top1 objective)
    C = int(point_idx.max()) + 1
    gpid = groups.astype(np.int64) * C + point_idx.astype(np.int64)
    uniq_gpid, inverse = np.unique(gpid, return_inverse=True)
    true_arc = np.full(len(uniq_gpid), -1, dtype=np.int64)
    true_arc[inverse[np.flatnonzero(y == 1)]] = arcs[y == 1]
    # remap point_idx to dense 0..n_points-1
    point_idx_dense = inverse

    print(f"optimising {args.model} on {X.shape} pairs, objective={args.objective}, "
          f"{args.trials} trials", flush=True)

    def objective(trial):
        _lgbm_space(trial)
        if args.objective == "top1":
            return _top1_objective(trial, X, y, groups, point_idx_dense,
                                   arcs, true_arc, _make_lgbm)
        return _macro_f1_objective(trial, X, y, groups, point_idx_dense,
                                   true_arc, _make_lgbm)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    print(f"optimisation done in {time.time()-t0:.0f}s", flush=True)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(RES, f"bayes_search_{args.model}_{args.objective}.csv"),
              index=False)

    best = study.best_params
    with open(os.path.join(RES, f"best_params_{args.model}_{args.objective}.json"),
              "w") as f:
        json.dump(dict(best=best, best_value=study.best_value,
                       objective=args.objective, trials=args.trials), f, indent=2)
    print(f"best params: {best}")
    print(f"best CV {args.objective}: {study.best_value:.4f}")

    est = _make_lgbm(best)
    est.fit(X, y, sample_weight=compute_sample_weights(y))
    out = os.path.join(ART, f"{args.model}_tuned.joblib")
    joblib.dump(est, out)
    print(f"saved tuned model to {out}")


if __name__ == "__main__":
    main()
