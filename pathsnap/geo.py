"""Geospatial helpers for PathSnap.

All distance calculations are performed in metres.  Because each trip is local to
a single city (a few kilometres across), we use a local equirectangular
("plate carree") projection centred on the trip centroid.  This is exact in the
north-south direction and only off by the cosine-of-latitude factor in the
east-west direction, which is constant over the small extent of a single city
and therefore introduces negligible relative error for *local* distances.

For the small number of cases where true geodesic distance is required (e.g.
reporting) we provide `haversine`.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6371008.8  # mean Earth radius (metres)


def haversine(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres between two (sets of) points.

    Accepts scalars or arrays (broadcast together).
    """
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_M * c


class LocalProjection:
    """Project lon/lat to a local metre grid centred on (clon, clat)."""

    def __init__(self, clon: float, clat: float):
        self.clon = float(clon)
        self.clat = float(clat)
        self._coslat = np.cos(np.radians(self.clat))
        # metres per degree of longitude / latitude at the centre
        self._mx = self._coslat * np.pi / 180.0 * EARTH_RADIUS_M
        self._my = np.pi / 180.0 * EARTH_RADIUS_M

    def xy(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = (lon - self.clon) * self._mx
        y = (lat - self.clat) * self._my
        return x, y

    def lonlat(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lon = x / self._mx + self.clon
        lat = y / self._my + self.clat
        return lon, lat


def bearing(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Initial bearing (degrees, [0, 360)) from point 1 to point 2."""
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    b = np.degrees(np.arctan2(y, x))
    return np.mod(b, 360.0)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest unsigned difference (degrees) between two bearings in [0, 180]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def point_segment(x, y, ax1, ay1, ax2, ay2):
    """Point-to-segment distance and projection parameter.

    Returns
    -------
    dist : ndarray (m)
        Perpendicular distance from (x, y) to the segment.
    t : ndarray
        Fractional projection along the segment, clipped to [0, 1].
    px, py : ndarray
        Closest point on the segment.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax1 = np.asarray(ax1, dtype=float)
    ay1 = np.asarray(ay1, dtype=float)
    ax2 = np.asarray(ax2, dtype=float)
    ay2 = np.asarray(ay2, dtype=float)
    dx = ax2 - ax1
    dy = ay2 - ay1
    length2 = dx * dx + dy * dy
    length2 = np.where(length2 == 0, 1e-12, length2)
    t = ((x - ax1) * dx + (y - ay1) * dy) / length2
    t = np.clip(t, 0.0, 1.0)
    px = ax1 + t * dx
    py = ay1 + t * dy
    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    return dist, t, px, py


def segment_bearing(ax1, ay1, ax2, ay2) -> np.ndarray:
    """Bearing (degrees [0,360)) of a segment in the local projected grid."""
    dx = np.asarray(ax2, dtype=float) - np.asarray(ax1, dtype=float)
    dy = np.asarray(ay2, dtype=float) - np.asarray(ay1, dtype=float)
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)
