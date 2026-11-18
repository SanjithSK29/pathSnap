"""Generate all report figures (comparison charts, confusion matrices, loss
curves, optimisation history) and Folium map screenshots.

Run:  PYTHONPATH=. python3 scripts/make_figures.py
"""
from __future__ import annotations

import os
import subprocess

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap.hmm import HMMDecoder
from pathsnap.io import load_trip
from scripts.generate_maps import build_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
MAP = os.path.join(ROOT, "report", "maps")
RES = os.path.join(ROOT, "experiments", "results")
ART = os.path.join(ROOT, "artifacts", "models")
EXP = os.path.join(ROOT, "experiments")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MAP, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 120})


def bar(csv_col, out, ylabel, color="#4C72B0", err_col=None, title=None):
    df = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    names = df["model"]
    vals = df[csv_col]
    err = df[err_col] if err_col else None
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    yerr = err.values if err is not None else None
    ax.bar(names, vals, yerr=yerr, color=color, edgecolor="white", capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(vals):
        ax.text(i, v + (err.values[i] if err is not None else 0), f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)


def confusion_matrix_figure(model_path, out):
    from pathsnap.evaluate import point_predictions
    from sklearn.metrics import confusion_matrix
    model = joblib.load(model_path)
    y_all, p_all = [], []
    for f in sorted(os.listdir(os.path.join(EXP, "student_test"))):
        d = np.load(os.path.join(EXP, "student_test", f))
        proba = model.predict_proba(d["X"])[:, 1]
        pred, _ = point_predictions(d["point_idx"], d["arcs"], proba, int(d["n_points"]))
        y_all.append(d["y"])
        p_all.append((proba >= 0.5).astype(int))
    y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
    cm = confusion_matrix(y_all, p_all, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["incorrect", "correct"])
    ax.set_yticklabels(["incorrect", "correct"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title("Pair-level confusion matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)


def mlp_loss_curve(out):
    path = os.path.join(ART, "MLP.joblib")
    if not os.path.exists(path):
        print("MLP model not found; skipping loss curve")
        return
    pipe = joblib.load(path)
    mlp = pipe.named_steps["mlpclassifier"]
    loss = np.array(mlp.loss_curve_)
    val = np.array(mlp.validation_scores_)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(loss, label="training loss")
    ax.plot(np.arange(1, len(val) + 1) * (len(loss) / len(val)), val,
            label="validation score", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / validation score")
    ax.set_title("MLP training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)


def bayes_history(out):
    for name in ("LightGBM_top1", "LightGBM", "LightGBM_macro_f1"):
        path = os.path.join(RES, f"bayes_search_{name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(range(len(df)), df["value"], marker="o", ms=4)
        ax.set_xlabel("trial")
        ax.set_ylabel("CV objective")
        ax.set_title(f"Bayesian optimisation history ({name.replace('_', ' ')})")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"bayes_history_{name}.png"))
        plt.close(fig)


def render(html, png):
    subprocess.run([
        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--window-size=1100,850",
        f"--screenshot={png}", f"file://{os.path.abspath(html)}",
    ], check=True, capture_output=True)


def prediction_maps(model_name="LightGBM", trips=("00000000", "00000056")):
    model_path = os.path.join(ART, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(ART, "LightGBM.joblib")
    model = joblib.load(model_path)
    for tid in trips:
        trip = load_trip(os.path.join(ROOT, "student_test"), tid)
        from pathsnap.pipeline import predict_trip
        res = predict_trip(model, trip, use_hmm=True, hmm=HMMDecoder())
        html = os.path.join(MAP, f"pred_{tid}.html")
        build_map(trip, res["final_arcs"], html, title=f"Trip {tid}")
        render(html, os.path.join(FIG, f"pred_{tid}.png"))
        print(f"  map {tid} rendered")


def main():
    print("comparison charts...")
    bar("cv_macro_f1", "cmp_cv_macro_f1.png", "CV Macro F1",
        color="#4C72B0", err_col="cv_macro_f1_std")
    bar("student_top1", "cmp_student_top1.png", "Student-test Top-1 accuracy",
        color="#55A868")
    bar("student_macro_f1", "cmp_student_macro_f1.png", "Student-test Macro F1",
        color="#DD8452")
    bar("student_log_loss", "cmp_logloss.png", "Student-test Log Loss",
        color="#C44E52")

    print("confusion matrix...")
    best = os.path.join(ART, "LightGBM.joblib")
    if os.path.exists(best):
        confusion_matrix_figure(best, "confusion_matrix_lightgbm.png")

    print("MLP loss curve...")
    mlp_loss_curve("mlp_loss_curve.png")

    print("bayes history...")
    bayes_history("bayes_history")

    print("prediction maps...")
    prediction_maps()

    print("done")


if __name__ == "__main__":
    main()
