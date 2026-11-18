"""Candidate road-segment generation for each GPS observation.

For each observation we collect nearby road segments (arcs) using a spatial index
over the local projected grid.

A plain KD-tree over arc *midpoints* misses long arcs: a point can lie directly on
a 300 m arc whose midpoint and endpoints are all farther than the search radius.
We therefore sample each arc at a fixed metre spacing (including both endpoints)
and index every sample point with its arc id.  Querying all sample points within
``radius`` of an observation is then guaranteed to find every arc that passes
within ``radius`` of it.

Candidates are ordered by true perpendicular distance to the segment and padded to
a fixed width K with ``-1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, point_segment


@dataclass
class CandidateSet:
    arcs: np.ndarray       # (N, K) arc ids, -1 padding
    dists: np.ndarray      # (N, K) perpendicular distance (m), inf padding
    n_real: np.ndarray     # (N,) number of real candidates


def _sample_arcs(nx, ny, arcs, step: float = 25.0):
    """Sample points along every arc; return (points, arc_ids)."""
    a1 = arcs[:, 0]
    a2 = arcs[:, 1]
    dx = nx[a2] - nx[a1]
    dy = ny[a2] - ny[a1]
    length = np.sqrt(dx * dx + dy * dy)
    n_samples = np.maximum(2, np.ceil(length / step).astype(np.int64) + 1)

    pts = []
    ids = []
    for ai in range(len(arcs)):
        n = int(n_samples[ai])
        frac = np.linspace(0.0, 1.0, n)
        x = nx[a1[ai]] + frac * dx[ai]
        y = ny[a1[ai]] + frac * dy[ai]
        pts.append(np.column_stack([x, y]))
        ids.append(np.full(n, ai, dtype=np.int64))
    pts = np.vstack(pts)
    ids = np.concatenate(ids)
    return pts, ids


def generate_candidates(trip, radius: float = 60.0, k: int = 10,
                        sample_step: float = 25.0) -> CandidateSet:
    proj = LocalProjection(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    nx, ny = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    px, py = proj.xy(trip.track[:, 0], trip.track[:, 1])

    a1 = trip.arcs[:, 0]
    a2 = trip.arcs[:, 1]

    sample_pts, sample_ids = _sample_arcs(nx, ny, trip.arcs, step=sample_step)
    tree = cKDTree(sample_pts)

    N = len(px)
    out_arcs = np.full((N, k), -1, dtype=np.int64)
    out_dists = np.full((N, k), np.inf)
    n_real = np.zeros(N, dtype=np.int64)

    hits = tree.query_ball_point(np.column_stack([px, py]), r=radius)

    for i in range(N):
        if not hits[i]:
            continue
        cand_ids = np.unique(sample_ids[hits[i]])
        if len(cand_ids) == 0:
            continue
        d, _, _, _ = point_segment(
            px[i], py[i], nx[a1[cand_ids]], ny[a1[cand_ids]],
            nx[a2[cand_ids]], ny[a2[cand_ids]],
        )
        order = np.argsort(d, kind="stable")
        cand_ids = cand_ids[order]
        d = d[order]
        keep = d <= radius
        cand_ids = cand_ids[keep]
        d = d[keep]
        m = min(len(cand_ids), k)
        out_arcs[i, :m] = cand_ids[:m]
        out_dists[i, :m] = d[:m]
        n_real[i] = m
    return CandidateSet(arcs=out_arcs, dists=out_dists, n_real=n_real)
