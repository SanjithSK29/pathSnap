"""Model catalogue and training helpers.

Six models spanning four algorithm families (linear/probabilistic, ensemble,
boosting, neural) are trained and compared, satisfying the minimum of five
distinct models required by the brief.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _make_logreg():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
    )


def _make_rf():
    return RandomForestClassifier(
        n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=0,
        min_samples_leaf=20, max_depth=20,
    )


def _make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.1, max_leaf_nodes=31, random_state=0,
        early_stopping=False,
    )


def _make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0,
        eval_metric="logloss",
    )


def _make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=250, num_leaves=31, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=0, verbose=-1,
    )


def _make_mlp():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", max_iter=100,
            early_stopping=True, n_iter_no_change=10, random_state=0,
        ),
    )


MODELS = [
    dict(name="Logistic Regression", family="Linear / probabilistic",
         key_config="C=1.0, L2, class_weight=balanced",
         make=_make_logreg, needs_scale=True),
    dict(name="Random Forest", family="Ensemble (bagging)",
         key_config="150 trees, max_depth=20, min_samples_leaf=20",
         make=_make_rf, needs_scale=False),
    dict(name="HistGradientBoosting", family="Boosting",
         key_config="250 iters, lr=0.1, max_leaf_nodes=31",
         make=_make_hgb, needs_scale=False),
    dict(name="XGBoost", family="Boosting",
         key_config="250 trees, max_depth=6, lr=0.1, hist",
         make=_make_xgb, needs_scale=False),
    dict(name="LightGBM", family="Boosting",
         key_config="250 trees, num_leaves=31, lr=0.1",
         make=_make_lgbm, needs_scale=False),
    dict(name="MLP", family="Neural network",
         key_config="hidden (64,32), ReLU, early stopping",
         make=_make_mlp, needs_scale=True),
]


def compute_sample_weights(y):
    """Balanced sample weights with mean 1.0 (equivalent to class_weight='balanced').

    Weights are n/(n_classes * n_class) so their sum equals n_samples; this keeps
    gradients on the same scale for sample_weight-based models (XGBoost/LightGBM/
    HistGradientBoosting).
    """
    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    n = float(len(y))
    w = np.ones(len(y), dtype=float)
    w[y == 1] = n / (2.0 * max(pos, 1.0))
    w[y == 0] = n / (2.0 * max(neg, 1.0))
    return w


def fit_model(spec, X, y, sample_weight=None):
    est = spec["make"]()
    t0 = time.time()
    name = spec["name"]
    if name in ("HistGradientBoosting", "XGBoost", "LightGBM"):
        est.fit(X, y, sample_weight=sample_weight)
    else:
        est.fit(X, y)
    return est, time.time() - t0
