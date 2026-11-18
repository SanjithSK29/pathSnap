"""Feature engineering for candidate road-segment scoring.

Each training/input example is a (GPS observation, candidate arc) pair.  We compute
observation-level motion features (heading, speed, time deltas) once per point, then
geometry features for each candidate segment.  The classifier scores each pair;
the pair with the highest score per observation is the predicted road segment.
"""
from __future__ import annotations

import numpy as np

from .geo import LocalProjection, angle_diff, bearing, haversine, point_segment

FEATURE_NAMES = [
    "dist_perp",          # perpendicular distance point->segment (m)
    "proj_frac",          # fractional projection along segment [0,1]
    "dist_from_node",     # distance to 'from' node (m)
    "dist_to_node",       # distance to 'to' node (m)
    "arc_len",            # segment length (m)
    "bearing_diff",       # min heading-vs-segment angle (deg)
    "signed_dist",        # signed perpendicular distance (m)
    "heading",            # GPS heading (deg)
    "speed",              # GPS speed (m/s)
    "dt_prev",            # time since previous observation (s)
    "dt_next",            # time until next observation (s)
    "dist_to_prev",       # distance to previous observation (m)
    "dist_to_next",       # distance to next observation (m)
    "dist_min_set",       # min perpendicular distance among the point's candidates
    "dist_rel",           # dist_perp - dist_min_set (relative distance)
]


def observation_features(trip) -> dict[str, np.ndarray]:
    """Per-observation motion features (shape (N,))."""
    lon = trip.track[:, 0]
    lat = trip.track[:, 1]
    t = trip.track[:, 2]
    N = len(lon)

    # heading via forward difference (last point uses backward difference)
    heading = np.empty(N)
    heading[:-1] = bearing(lon[:-1], lat[:-1], lon[1:], lat[1:])
    heading[-1] = heading[-2] if N > 1 else 0.0

    # speed via forward difference (m/s)
    dist_fwd = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dt_fwd = np.diff(t)
    dt_fwd = np.where(np.abs(dt_fwd) < 1e-9, 1e-9, dt_fwd)
    speed = np.empty(N)
    speed[:-1] = dist_fwd / np.abs(dt_fwd)
    speed[-1] = speed[-2] if N > 1 else 0.0

    dt_prev = np.empty(N)
    dt_prev[0] = 1.0
    dt_prev[1:] = np.abs(np.diff(t))
    dt_next = np.empty(N)
    dt_next[:-1] = np.abs(np.diff(t))
    dt_next[-1] = 1.0

    dist_to_prev = np.empty(N)
    dist_to_prev[0] = dist_fwd[0] if N > 1 else 0.0
    dist_to_prev[1:] = dist_fwd
    dist_to_next = np.empty(N)
    dist_to_next[:-1] = dist_fwd
    dist_to_next[-1] = dist_fwd[-1] if N > 1 else 0.0

    return dict(
        heading=heading, speed=speed, dt_prev=dt_prev, dt_next=dt_next,
        dist_to_prev=dist_to_prev, dist_to_next=dist_to_next,
    )


def pair_features(trip, cand: "CandidateSet") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the flat feature matrix over *valid* (point, candidate) pairs.

    Returns
    -------
    X : (M, F) float
    point_idx : (M,) int64
    arcs : (M,) int64
    """
    proj = LocalProjection(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    nx, ny = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    px, py = proj.xy(trip.track[:, 0], trip.track[:, 1])

    obs = observation_features(trip)
    heading = obs["heading"]
    speed = obs["speed"]
    dt_prev = obs["dt_prev"]
    dt_next = obs["dt_next"]
    dist_to_prev = obs["dist_to_prev"]
    dist_to_next = obs["dist_to_next"]

    arcs = cand.arcs
    N, K = arcs.shape

    valid_mask = arcs >= 0
    rows = np.nonzero(valid_mask)
    point_idx = rows[0].astype(np.int64)
    a = arcs[valid_mask].astype(np.int64)

    a1 = trip.arcs[a, 0]
    a2 = trip.arcs[a, 1]
    ax1, ay1, ax2, ay2 = nx[a1], ny[a1], nx[a2], ny[a2]

    pi = point_idx
    x = px[pi]
    y = py[pi]

    dist_perp, t, qx, qy = point_segment(x, y, ax1, ay1, ax2, ay2)

    dx = ax2 - ax1
    dy = ay2 - ay1
    length = np.sqrt(dx * dx + dy * dy)
    # signed distance via cross product
    cross = (x - ax1) * dy - (y - ay1) * dx
    signed = cross / np.where(length < 1e-9, 1e-9, length)

    seg_bearing = np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)
    bdiff = angle_diff(heading[pi], seg_bearing)

    dist_from = np.sqrt((x - ax1) ** 2 + (y - ay1) ** 2)
    dist_to = np.sqrt((x - ax2) ** 2 + (y - ay2) ** 2)

    X = np.column_stack([
        dist_perp,
        t,
        dist_from,
        dist_to,
        length,
        bdiff,
        signed,
        heading[pi],
        speed[pi],
        dt_prev[pi],
        dt_next[pi],
        dist_to_prev[pi],
        dist_to_next[pi],
    ])
    return X, point_idx, a


def candidate_projection(trip, cand_arcs):
    """Projection fraction t (0..1) of each candidate arc for each point.

    Returns (N, K) float32; -1 for padding entries.
    """
    proj = LocalProjection(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    nx, ny = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    px, py = proj.xy(trip.track[:, 0], trip.track[:, 1])

    N, K = cand_arcs.shape
    out = np.full((N, K), -1.0, dtype=np.float32)
    valid = cand_arcs >= 0
    a = cand_arcs[valid].astype(np.int64)
    pi = np.nonzero(valid)[0]
    a1 = trip.arcs[a, 0]
    a2 = trip.arcs[a, 1]
    _, t, _, _ = point_segment(px[pi], py[pi], nx[a1], ny[a1], nx[a2], ny[a2])
    out[valid] = t
    return out


def pair_features_with_relative(trip, cand):
    """Full feature matrix (all 15 features) over valid (point, candidate) pairs.

    Returns (X, point_idx, arcs), matching what the models were trained on.
    """
    X, point_idx, arcs = pair_features(trip, cand)
    dists = cand.dists
    cand_matrix = cand.arcs
    min_dist = np.where(cand_matrix >= 0, dists, np.inf).min(axis=1)
    dist_min_set = min_dist[point_idx]
    dist_rel = X[:, 0] - dist_min_set
    X = np.column_stack([X, dist_min_set, dist_rel]).astype(np.float32)
    return X, point_idx, arcs
