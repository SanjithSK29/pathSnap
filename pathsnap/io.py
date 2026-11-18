"""I/O and data-structure helpers for PathSnap trips."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N, 3)  lon, lat, t(seconds)
    nodes: np.ndarray          # (V, 2)  lon, lat
    arcs: np.ndarray           # (E, 2)  from_node, to_node (int)
    route: np.ndarray | None   # (R,)    arc_id sequence, or None if absent

    def __post_init__(self):
        self.arcs = self.arcs.astype(np.int64)
        if self.route is not None:
            self.route = self.route.astype(np.int64)

    @property
    def n_points(self) -> int:
        return int(self.track.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])


def load_trip(directory: str, trip_id: str) -> Trip:
    """Load the four raw files of a trip from `directory/trip_id/`."""
    base = os.path.join(directory, trip_id, trip_id)

    track = np.loadtxt(base + ".track", ndmin=2)
    if track.shape[1] == 1:  # single-row guard
        track = track.reshape(-1, 3)
    nodes = np.loadtxt(base + ".nodes", ndmin=2)
    arcs = np.loadtxt(base + ".arcs", ndmin=2, dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(-1, 2)

    route_path = base + ".route"
    if os.path.exists(route_path):
        route = np.loadtxt(route_path, ndmin=1, dtype=np.int64)
        route = np.atleast_1d(route)
    else:
        route = None

    return Trip(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def list_trips(directory: str) -> list[str]:
    """Return sorted trip ids present in `directory`."""
    names = sorted(
        n for n in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, n))
    )
    return names
