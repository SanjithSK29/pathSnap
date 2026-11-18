"""Generate interactive Folium maps comparing GPS / reference / predicted routes.

Run:  PYTHONPATH=. python3 scripts/generate_maps.py --model LightGBM_tuned
      --trip 00000000 --out report/maps/pred_00000000.html
"""
from __future__ import annotations

import argparse
import os
import subprocess

import joblib
import numpy as np

from pathsnap.hmm import HMMDecoder
from pathsnap.io import load_trip
from pathsnap.pipeline import predict_trip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "models")


def build_map(trip, pred_arcs, out_html, title=None, show_network=True):
    import folium
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14,
                   tiles="OpenStreetMap")

    # road-network geometry (arcs near the trajectory bounding box)
    if show_network:
        fg_net = folium.FeatureGroup(name="Road network", show=False)
        pad = 0.002
        mask = ((trip.nodes[:, 0] >= lon.min() - pad) & (trip.nodes[:, 0] <= lon.max() + pad)
                & (trip.nodes[:, 1] >= lat.min() - pad) & (trip.nodes[:, 1] <= lat.max() + pad))
        keep_arc = mask[trip.arcs[:, 0]] & mask[trip.arcs[:, 1]]
        idx = np.flatnonzero(keep_arc)
        step = max(1, len(idx) // 4000)
        segs = []
        for ai in idx[::step]:
            u, v = trip.arcs[ai]
            segs.append([[trip.nodes[u][1], trip.nodes[u][0]],
                         [trip.nodes[v][1], trip.nodes[v][0]]])
        folium.PolyLine(segs, color="#9aa7b8", weight=1, opacity=0.4).add_to(fg_net)
        m.add_child(fg_net)

    fg_gps = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(lon) // 2000)
    for i in range(0, len(lon), step):
        folium.CircleMarker([lat[i], lon[i]], radius=1.5, color="blue",
                            fill=True, fill_opacity=0.8).add_to(fg_gps)

    # predicted route
    fg_pred = folium.FeatureGroup(name="Predicted route")
    pts = _arc_polyline(trip, pred_arcs)
    folium.PolyLine(pts, color="red", weight=4, opacity=0.9).add_to(fg_pred)

    # reference route
    fg_ref = folium.FeatureGroup(name="Reference route")
    if trip.route is not None:
        pts = _arc_polyline(trip, trip.route)
        folium.PolyLine(pts, color="green", weight=3, opacity=0.9).add_to(fg_ref)

    for fg in (fg_gps, fg_pred, fg_ref):
        m.add_child(fg)
    folium.LayerControl().add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(f"<h3 style='margin:4px'>{title}</h3>"))
    m.save(out_html)
    return out_html


def _arc_polyline(trip, arc_ids):
    pts = []
    prev_end = None
    for a in arc_ids:
        u, v = trip.arcs[a]
        p1 = (trip.nodes[u][1], trip.nodes[u][0])
        p2 = (trip.nodes[v][1], trip.nodes[v][0])
        if pts and prev_end == p1:
            pts.append(p2)
        elif pts and prev_end == p2:
            pts.append(p1)
        else:
            pts.append(p1)
            pts.append(p2)
        prev_end = pts[-1]
    return pts


def render_png(html_path, png_path):
    subprocess.run([
        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--window-size=1100,850",
        f"--screenshot={png_path}", f"file://{os.path.abspath(html_path)}",
    ], check=True, capture_output=True)
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LightGBM_tuned")
    ap.add_argument("--trip", default="00000000")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "student_test"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = joblib.load(os.path.join(ART, f"{args.model}.joblib"))
    trip = load_trip(args.data_dir, args.trip)
    res = predict_trip(model, trip, use_hmm=True, hmm=HMMDecoder())

    out = args.out or os.path.join(ROOT, "report", "maps", f"pred_{args.trip}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_map(trip, res["final_arcs"], out, title=f"Trip {args.trip}")
    print("saved", out)


if __name__ == "__main__":
    main()
