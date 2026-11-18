"""Exploratory data analysis.

Generates the EDA outputs required by the brief:
  * dataset inventory + summary statistics (CSV),
  * trajectory / sampling-interval / network-size distributions (PNG),
  * interactive Folium maps of representative trips (HTML).

Run:  PYTHONPATH=. python3 scripts/eda.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap.geo import haversine
from pathsnap.io import list_trips, load_trip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
MAP = os.path.join(ROOT, "report", "maps")
RES = os.path.join(ROOT, "experiments", "results")
for d in (FIG, MAP, RES):
    os.makedirs(d, exist_ok=True)


def _connected_components(n_nodes, arcs):
    """Number of connected components and isolated nodes in the road network."""
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for u, v in arcs:
        union(int(u), int(v))
    roots = {find(i) for i in range(n_nodes)}
    degree = np.zeros(n_nodes, dtype=int)
    for u, v in arcs:
        degree[int(u)] += 1
        degree[int(v)] += 1
    isolated = int((degree == 0).sum())
    return len(roots), isolated


def trip_statistics(directory, tid):
    import os
    trip = load_trip(directory, tid)
    lon, lat, t = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]
    dt = np.abs(np.diff(t))
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    n = trip.n_points
    base = os.path.join(directory, tid, tid)
    sizes = {ext: os.path.getsize(base + ext) for ext in (".track", ".nodes", ".arcs", ".route")}
    n_components, isolated = _connected_components(trip.n_nodes, trip.arcs)
    return dict(
        trip_id=tid,
        n_points=n,
        n_nodes=trip.n_nodes,
        n_arcs=trip.n_arcs,
        route_len=len(trip.route) if trip.route is not None else np.nan,
        track_bytes=sizes[".track"], nodes_bytes=sizes[".nodes"],
        arcs_bytes=sizes[".arcs"], route_bytes=sizes[".route"],
        lon_min=lon.min(), lon_max=lon.max(),
        lat_min=lat.min(), lat_max=lat.max(),
        lon_mean=lon.mean(), lat_mean=lat.mean(),
        dt_1s_frac=float((np.abs(dt - 1.0) < 1e-6).mean()),
        dt_median=float(np.median(dt)),
        dt_max=float(dt.max()),
        step_median=float(np.median(step)),
        step_p99=float(np.percentile(step, 99)),
        step_max=float(step.max()),
        arc_density=float(trip.n_arcs / max(n, 1)),
        n_components=int(n_components),
        isolated_nodes=int(isolated),
    )


def build_inventory():
    rows = []
    for part in ("train", "student_test"):
        for tid in list_trips(os.path.join(ROOT, part)):
            r = trip_statistics(os.path.join(ROOT, part), tid)
            r["partition"] = part
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "dataset_inventory.csv"), index=False)
    return df


def make_plots(df):
    plt.rcParams.update({"font.size": 9})

    # 1. sampling interval + trajectory length
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(df["n_points"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("GPS observations per trip")
    axes[0].set_ylabel("trips")
    axes[0].set_title("Trajectory length distribution")
    axes[1].hist(df["dt_1s_frac"], bins=40, color="#DD8452", edgecolor="white")
    axes[1].set_xlabel("fraction of 1 s steps")
    axes[1].set_ylabel("trips")
    axes[1].set_title("Sampling regularity")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_trajectory.png"), dpi=160)
    plt.close(fig)

    # 2. network size
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(df["n_nodes"], bins=40, color="#55A868", edgecolor="white")
    axes[0].set_xlabel("road-network nodes")
    axes[0].set_ylabel("trips")
    axes[0].set_title("Network node count")
    axes[1].hist(df["n_arcs"], bins=40, color="#C44E52", edgecolor="white")
    axes[1].set_xlabel("road-network arcs")
    axes[1].set_title("Network arc count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_network_size.png"), dpi=160)
    plt.close(fig)

    # 3. step length (GPS noise / speed proxy)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(df["step_median"], bins=40, color="#8172B3", edgecolor="white")
    axes[0].set_xlabel("median step length (m)")
    axes[0].set_ylabel("trips")
    axes[0].set_title("Median GPS step length")
    axes[1].hist(df["dt_max"], bins=40, color="#8C564B", edgecolor="white")
    axes[1].set_xlabel("max time gap (s)")
    axes[1].set_title("Largest within-trip time gap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_steps_gaps.png"), dpi=160)
    plt.close(fig)

    # 4. global coordinate distribution
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(df["lon_mean"], df["lat_mean"], s=12, c="tab:blue", alpha=0.7)
    ax.set_xlabel("mean longitude")
    ax.set_ylabel("mean latitude")
    ax.set_title("Geographical spread of the 90 trips")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_global_spread.png"), dpi=160)
    plt.close(fig)


def make_folium(trip_id, directory, out):
    import folium
    trip = load_trip(directory, trip_id)
    lon = trip.track[:, 0]
    lat = trip.track[:, 1]
    center = [lat.mean(), lon.mean()]

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    fg_track = folium.FeatureGroup(name="GPS observations")
    for i in range(0, len(lon), max(1, len(lon) // 2000)):
        folium.CircleMarker([lat[i], lon[i]], radius=1.5, color="blue",
                            fill=True, fill_opacity=0.8).add_to(fg_track)

    # reference route
    if trip.route is not None:
        fg_route = folium.FeatureGroup(name="Reference route")
        pts = []
        for ai in trip.route:
            u, v = trip.arcs[ai]
            pts.append((trip.nodes[u][1], trip.nodes[u][0]))
            pts.append((trip.nodes[v][1], trip.nodes[v][0]))
        pts = pts[::2]
        if pts:
            folium.PolyLine(pts, color="green", weight=3, opacity=0.9).add_to(fg_route)
        m.add_child(fg_route)

    m.add_child(fg_track)
    folium.LayerControl().add_to(m)
    m.save(out)
    print(f"  saved map {out}")


def main():
    print("building inventory...")
    df = build_inventory()
    print(f"  {len(df)} trips")
    print("  train trips:", (df.partition == "train").sum(),
          " student_test trips:", (df.partition == "student_test").sum())
    print("  total observations:", df.n_points.sum())
    print("  total nodes:", df.n_nodes.sum(), " total arcs:", df.n_arcs.sum())
    print("\nsummary stats:")
    print(df[["n_points", "n_nodes", "n_arcs", "dt_1s_frac", "step_median"]].describe().round(3).to_string())

    make_plots(df)

    print("\ngenerating folium maps...")
    for tid in ("00000083", "00000021"):
        make_folium(tid, os.path.join(ROOT, "train"), os.path.join(MAP, f"map_{tid}.html"))
    for tid in ("00000000", "00000056"):
        make_folium(tid, os.path.join(ROOT, "student_test"), os.path.join(MAP, f"map_{tid}.html"))


if __name__ == "__main__":
    main()
