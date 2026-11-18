"""Evaluation metrics for PathSnap.

Two levels are reported:

* **point-level** — for each observation, the predicted arc is the candidate with
  the highest score; ``top1_accuracy`` and ``miss_rate`` are point-level.
* **pair-level** — each (observation, candidate) pair is a binary decision
  (correct / not); ``macro_f1``, ``precision``, ``recall`` and ``log_loss`` are
  pair-level, matching the deliverable's request for a binary Macro F1 and Log Loss.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def point_predictions(point_idx, arcs, scores, n_points):
    """Argmax candidate per point.

    Returns (pred_arcs (n_points,), best_scores (n_points,)).
    """
    scores = np.asarray(scores, dtype=float)
    # primary sort by point, secondary by descending score
    order = np.lexsort((-scores, point_idx))
    idx = point_idx[order]
    a = arcs[order]
    s = scores[order]
    first = np.empty(len(idx), dtype=bool)
    first[0] = True
    first[1:] = idx[1:] != idx[:-1]
    pred = np.full(n_points, -1, dtype=np.int64)
    best_score = np.full(n_points, -np.inf)
    pred[idx[first]] = a[first]
    best_score[idx[first]] = s[first]
    return pred, best_score


def point_top1(point_idx, arcs, scores, true_arc, n_points):
    pred, _ = point_predictions(point_idx, arcs, scores, n_points)
    mask = true_arc >= 0
    return float((pred[mask] == true_arc[mask]).mean())


def evaluate_pairs(y_true, y_prob):
    """Pair-level metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                 zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return dict(precision=float(p), recall=float(r), f1=float(f),
                macro_f1=float(macro_f1), log_loss=float(ll), confusion_matrix=cm)


def route_continuity(pred_arcs, net_arcs):
    """Fraction of consecutive predicted arcs that are connected (equal or share a node).

    `net_arcs` is the (E, 2) array of from/to node ids.
    """
    if len(pred_arcs) < 2:
        return 1.0
    a = pred_arcs[:-1]
    b = pred_arcs[1:]
    conn = np.zeros(len(a), dtype=bool)
    valid = (a >= 0) & (b >= 0)
    if valid.any():
        aa = a[valid].astype(np.int64)
        bb = b[valid].astype(np.int64)
        # two arcs are connected if they share a node (or are equal)
        same = aa == bb
        nodes_a = net_arcs[aa]   # (m, 2)
        nodes_b = net_arcs[bb]   # (m, 2)
        share = ((nodes_a[:, 0] == nodes_b[:, 0]) | (nodes_a[:, 0] == nodes_b[:, 1])
                 | (nodes_a[:, 1] == nodes_b[:, 0]) | (nodes_a[:, 1] == nodes_b[:, 1]))
        conn[valid] = same | share
    return float(conn.mean())
