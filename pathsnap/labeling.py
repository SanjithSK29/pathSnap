"""Ground-truth labelling.

The reference `.route` file only gives the *sequence* of arc ids traversed by the
vehicle; it does not say which GPS observation belongs to which arc.  We derive an
observation-level label by aligning the GPS sequence to the reference route with a
monotone dynamic program (a restricted Viterbi over route *positions*):

* emission cost  = perpendicular distance (m) from the observation to the route arc,
* transition     = stay on the arc or skip forward by up to ``max_skip`` arcs.

Monotonicity reflects the fact that a vehicle traverses the route in order; the
skip allowance absorbs GPS gaps.  Route *revisits* are handled naturally because
each occurrence of an arc is a distinct route position.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .geo import LocalProjection, point_segment


def reference_route_geometry(trip):
    """Projected route node coordinates per route arc + the route arc ids."""
    proj = LocalProjection(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    nx, ny = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    route = trip.route
    a1 = trip.arcs[route, 0]
    a2 = trip.arcs[route, 1]
    return nx[a1], ny[a1], nx[a2], ny[a2], route


def _emission_matrix(px, py, ax1, ay1, ax2, ay2):
    """D[i, j] = perpendicular distance (m) from point i to route arc j."""
    N = len(px)
    R = len(ax1)
    D = np.empty((N, R), dtype=np.float32)
    chunk = 256
    for i in range(0, N, chunk):
        idx = np.arange(i, min(i + chunk, N))
        pxb = px[idx][:, None]
        pyb = py[idx][:, None]
        dx = ax2[None, :] - ax1[None, :]
        dy = ay2[None, :] - ay1[None, :]
        L2 = dx * dx + dy * dy
        L2 = np.where(L2 == 0, 1e-12, L2)
        t = ((pxb - ax1[None, :]) * dx + (pyb - ay1[None, :]) * dy) / L2
        t = np.clip(t, 0.0, 1.0)
        qx = ax1[None, :] + t * dx
        qy = ay1[None, :] + t * dy
        D[idx] = np.sqrt((pxb - qx) ** 2 + (pyb - qy) ** 2)
    return D


def _window_argmin(dp, window):
    """For each j, argmin of dp over [j-window+1, j] (backward-inclusive)."""
    n = len(dp)
    best = np.empty(n, dtype=np.int32)
    dq = deque()
    for j in range(n):
        while dq and dp[dq[-1]] >= dp[j]:
            dq.pop()
        dq.append(j)
        while dq and dq[0] <= j - window:
            dq.popleft()
        best[j] = dq[0]
    return best


def label_trip(trip, max_skip: int = 20):
    """Assign each observation to a reference route arc id.

    Returns
    -------
    labels : (N,) int64  arc_id per observation
    dists  : (N,) float  perpendicular distance to the assigned arc (m)
    """
    proj = LocalProjection(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    px, py = proj.xy(trip.track[:, 0], trip.track[:, 1])
    nx, ny = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    ax1, ay1, ax2, ay2, route = reference_route_geometry(trip)
    R = len(route)
    N = len(px)

    D = _emission_matrix(px, py, ax1, ay1, ax2, ay2)

    # Forward Viterbi with back-pointers.
    dp = D[0].copy()
    backptr = np.zeros((N, R), dtype=np.int32)
    for i in range(1, N):
        best = _window_argmin(dp, max_skip + 1)
        backptr[i] = best
        dp = D[i] + dp[best]

    # Traceback.
    path = np.empty(N, dtype=np.int32)
    path[-1] = int(np.argmin(dp))
    for i in range(N - 2, -1, -1):
        path[i] = backptr[i + 1][path[i + 1]]

    labels = route[path].astype(np.int64)
    a1 = trip.arcs[labels, 0]
    a2 = trip.arcs[labels, 1]
    d, _, _, _ = point_segment(px, py, nx[a1], ny[a1], nx[a2], ny[a2])
    return labels, d
