"""Dataset assembly: candidates -> features -> pair labels, cached to disk."""
from __future__ import annotations

import os

import numpy as np

from .candidates import generate_candidates
from .features import FEATURE_NAMES, pair_features_with_relative
from .io import load_trip
from .labeling import label_trip


def build_trip_pairs(trip, radius: float, k: int):
    """Build flat (point, candidate) pairs for one trip.

    Returns
    -------
    X : (M, F) float32
    point_idx : (M,) int32   (within-trip point index)
    arcs : (M,) int32
    y : (M,) uint8          1 if candidate == reference arc
    true_arc : (N,) int64   reference arc per point
    miss_mask : (N,) bool   True where the reference arc is absent from candidates
    """
    cand = generate_candidates(trip, radius=radius, k=k)
    X, point_idx, arcs = pair_features_with_relative(trip, cand)
    true_arc, _ = label_trip(trip)

    cand_matrix = cand.arcs
    y = (arcs == true_arc[point_idx]).astype(np.uint8)

    # miss mask: true arc not present in candidate matrix for that point
    miss_mask = ~(cand_matrix == true_arc[:, None]).any(axis=1)

    return (
        X.astype(np.float32),
        point_idx.astype(np.int32),
        arcs.astype(np.int32),
        y,
        true_arc.astype(np.int64),
        miss_mask,
    )


def build_dataset(
    directory: str,
    trips: list[str],
    radius: float,
    k: int,
    cache_path: str | None = None,
    verbose: bool = True,
):
    """Build and (optionally) cache the training dataset across all `trips`."""
    Xs, ps, ars, ys, trs, tris = [], [], [], [], [], []
    miss_total = 0
    n_total = 0

    for ti, tid in enumerate(trips):
        trip = load_trip(directory, tid)
        X, point_idx, arcs, y, true_arc, miss_mask = build_trip_pairs(trip, radius, k)
        Xs.append(X)
        ps.append(point_idx)
        ars.append(arcs)
        ys.append(y)
        trs.append(true_arc)
        tris.append(np.full(len(y), ti, dtype=np.int32))
        miss_total += int(miss_mask.sum())
        n_total += len(true_arc)
        if verbose and (ti + 1) % 10 == 0:
            print(f"  built {ti + 1}/{len(trips)} trips", flush=True)

    data = dict(
        X=np.concatenate(Xs),
        point_idx=np.concatenate(ps),
        arcs=np.concatenate(ars),
        y=np.concatenate(ys),
        trip_idx=np.concatenate(tris),
        feature_names=np.array(FEATURE_NAMES),
        n_points=n_total,
        miss_total=miss_total,
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **data)
    return data
