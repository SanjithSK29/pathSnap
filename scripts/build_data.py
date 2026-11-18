"""Build and cache the training dataset and per-trip student-test structures."""
from __future__ import annotations

import os
import time

import numpy as np

from pathsnap.io import list_trips, load_trip
from pathsnap.dataset import build_dataset, build_trip_pairs
from pathsnap.candidates import generate_candidates
from pathsnap.labeling import label_trip

RADIUS = 60.0
K = 10

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")


def build_train(radius=60.0, k=10):
    trips = list_trips(os.path.join(ROOT, "train"))
    cache = os.path.join(EXP, "train_pairs.npz")
    t0 = time.time()
    data = build_dataset(os.path.join(ROOT, "train"), trips, radius, k, cache_path=cache)
    print(f"train dataset: {len(data['y'])} pairs, {data['n_points']} points, "
          f"miss {data['miss_total']} ({data['miss_total']/data['n_points']:.4f}), "
          f"pos rate {data['y'].mean():.4f}, time {time.time()-t0:.1f}s")
    return cache


def build_student_test(radius=60.0, k=10):
    trips = list_trips(os.path.join(ROOT, "student_test"))
    outdir = os.path.join(EXP, "student_test")
    os.makedirs(outdir, exist_ok=True)
    for t in trips:
        path = os.path.join(outdir, f"{t}.npz")
        if os.path.exists(path):
            continue
        trip = load_trip(os.path.join(ROOT, "student_test"), t)
        cand = generate_candidates(trip, radius=radius, k=k)
        from pathsnap.features import candidate_projection, pair_features_with_relative
        X, point_idx, arcs = pair_features_with_relative(trip, cand)
        cand_proj = candidate_projection(trip, cand.arcs)
        true_arc, _ = label_trip(trip)
        y = (arcs == true_arc[point_idx]).astype(np.uint8)
        np.savez_compressed(
            path,
            X=X.astype(np.float32),
            point_idx=point_idx.astype(np.int32),
            arcs=arcs.astype(np.int32),
            y=y,
            true_arc=true_arc.astype(np.int64),
            n_points=trip.n_points,
            track=trip.track.astype(np.float32),
            nodes=trip.nodes.astype(np.float32),
            net_arcs=trip.arcs.astype(np.int64),
            route=trip.route.astype(np.int64),
            cand_arcs=cand.arcs.astype(np.int32),
            cand_proj=cand_proj.astype(np.float32),
        )
    print(f"student_test: cached {len(trips)} trips in {outdir}")
    return outdir


if __name__ == "__main__":
    build_train(RADIUS, K)
    build_student_test(RADIUS, K)
