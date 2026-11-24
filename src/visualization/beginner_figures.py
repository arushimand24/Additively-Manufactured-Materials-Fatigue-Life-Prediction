"""Figures for the beginner's guide (reports/Understanding_Our_Project.docx).

Every number drawn here is read from the project's saved outputs; nothing is retrained:
    results/metrics/metrics_by_model_and_scheme.csv      RMSE per model and validation scheme
    results/metrics/bootstrap_intervals.csv              95% article-bootstrap intervals (conditional on the fitted models)
    results/predictions/oof_predictions_primary.csv      out-of-fold predictions of the primary experiment
Figure 2 is an explanatory diagram with an invented, clearly labelled example; it shows no project data.

Usage (from the project root):  python src/visualization/beginner_figures.py
or from notebook 04:            from visualization.beginner_figures import make_all; make_all(PROJECT_ROOT)
Each figure is saved as a 300-dpi PNG and as a vector PDF in results/figures/beginner/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # inside notebook 04 keep the notebook's own backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SCHEMES = ["A_random", "B_curve", "C_article"]
SCHEME_NAMES = {"A_random": "Random points", "B_curve": "Whole curves\nheld out", "C_article": "Whole articles\nheld out"}
SCHEME_COLORS = {"A_random": "#2a78d6", "B_curve": "#eb6834", "C_article": "#1baf7a"}
MODELS = ["Basquin", "Ridge", "RandomForest", "LightGBM", "BasquinResidual"]
MODEL_NAMES = {"Basquin": "Basquin line\n(per alloy)", "Ridge": "Ridge\nregression", "RandomForest": "Random\nForest",
               "LightGBM": "LightGBM", "BasquinResidual": "Basquin residual\nhybrid"}
INK, MUTED, GRID = "#2f2e2b", "#6b6a66", "#e6e4df"
TRAIN, TEST = "#2a78d6", "#eb6834"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "axes.edgecolor": "#b5b3ac", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK, "ytick.color": INK, "legend.frameon": False, "savefig.dpi": 300,
    })


def load(root: Path) -> dict:
    root = Path(root)
    metrics = pd.read_csv(root / "results" / "metrics" / "metrics_by_model_and_scheme.csv")
    boot = pd.read_csv(root / "results" / "metrics" / "bootstrap_intervals.csv")
    pred = pd.read_csv(root / "results" / "predictions" / "oof_predictions_primary.csv",
                       usecols=["row_id", "article_id", "scheme", "model", "y_true", "y_pred"])
    return {"metrics": metrics, "boot": boot, "pred": pred}


def figure_values(data: dict) -> dict:
    """The numbers the figures show, taken straight from the saved files (also used by the tests and the guide)."""
    m = data["metrics"].set_index(["model", "scheme"])
    b = data["boot"]
    rmse = {(mo, s): float(m.loc[(mo, s), "rmse"]) for mo in MODELS + ["MeanPredictor"] for s in SCHEMES}
    ci = {(r.model, r.scheme): (float(r.ci_low), float(r.ci_high)) for r in b[b["quantity"] == "rmse"].itertuples()}
    infl = {(r.model, r.scheme): (float(r.estimate), float(r.ci_low), float(r.ci_high)) for r in b[b["quantity"] == "inflation_vs_random"].itertuples()}
    within = {(mo, s): float(m.loc[(mo, s), "within_0.3"]) for mo in MODELS for s in SCHEMES}
    return {"rmse": rmse, "ci": ci, "inflation": infl, "within_0.3": within}


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------------- Figure 1
def figure_1_central_finding(values: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(3)
    heights = [values["rmse"][("LightGBM", s)] for s in SCHEMES]
    lows = [values["rmse"][("LightGBM", s)] - values["ci"][("LightGBM", s)][0] for s in SCHEMES]
    highs = [values["ci"][("LightGBM", s)][1] - values["rmse"][("LightGBM", s)] for s in SCHEMES]
    ax.bar(x, heights, width=0.58, color=[SCHEME_COLORS[s] for s in SCHEMES], zorder=2)
    ax.errorbar(x, heights, yerr=[lows, highs], fmt="none", ecolor=INK, elinewidth=1.2, capsize=5, zorder=3)
    for xi, h in zip(x, heights):
        ax.text(xi, h / 2, f"{h:.3f}", ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    mean_c = values["rmse"][("MeanPredictor", "C_article")]
    ax.axhline(mean_c, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(-0.42, mean_c + 0.012, f"always predicting the average life: {mean_c:.3f}", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    ratio = heights[2] / heights[0]
    ax.annotate("", xy=(2, 1.08), xytext=(0, 1.08), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, connectionstyle="arc3,rad=-0.12"))
    ax.text(1.0, 1.17, f"error is {100 * (ratio - 1):.0f}% higher on unseen articles  ({heights[2]:.3f} / {heights[0]:.3f} = {ratio:.2f})",
            ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(x, [SCHEME_NAMES[s] for s in SCHEMES], fontsize=10.5)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("RMSE of log10(fatigue life)  —  lower is better")
    ax.set_title("Same model, same data: the error grows when whole research articles are unseen", fontsize=11.5, pad=12)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.035, "LightGBM, primary experiment: 12,666 failure points from 440 articles. Error bars: 95% intervals from resampling whole articles\n"
             "(conditional on the fitted models). An RMSE of 0.3 corresponds to about a factor of 2 in life for a single prediction.",
             fontsize=7.8, color=MUTED, ha="left", va="top")
    return _save(fig, out_dir, "fig1_central_finding")


# ----------------------------------------------------------------------------------------------- Figure 2
def figure_2_validation_strategies(out_dir: Path) -> list[Path]:
    """Explanatory diagram. Invented example: 3 papers x 2 curves x 5 measurements. Not project data."""
    test_sets = {
        "A": {(0, 0, 1), (0, 1, 3), (1, 0, 4), (1, 1, 0), (2, 0, 2), (2, 1, 4)},          # scattered points
        "B": {(1, 1, k) for k in range(5)},                                                  # one whole curve
        "C": {(2, c, k) for c in range(2) for k in range(5)},                                # one whole paper
    }
    titles = {
        "A": "A.  Random point-level split",
        "B": "B.  Curve-held-out split",
        "C": "C.  Article-held-out split",
    }
    notes = {
        "A": "Each tested measurement\nhas its neighbours on the\nsame curve, from the same\npaper, in the training data.",
        "B": "The tested curve is new,\nbut the other curve of the\nsame paper (same lab,\nmachine, powder) is still\nin the training data.",
        "C": "Nothing from Paper 3 is\nin the training data. This\nis closest to a new lab\nusing the model.",
    }
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.6))
    for ax, key in zip(axes, ["A", "B", "C"]):
        ax.set_xlim(0, 13.4)
        ax.set_ylim(0, 3.0)
        ax.axis("off")
        ax.text(0.0, 2.95, titles[key], fontsize=15, fontweight="bold", color=INK, va="top")
        for p in range(3):
            x0 = 0.1 + p * 3.05
            whole_test = key == "C" and p == 2
            box = FancyBboxPatch((x0, 0.15), 2.8, 2.25, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 fc="#fdf1ea" if whole_test else "#f6f5f2", ec=TEST if whole_test else "#c9c7c0", lw=1.6 if whole_test else 1.0)
            ax.add_patch(box)
            ax.text(x0 + 1.4, 2.2, f"Paper {p + 1}" + ("  (tested)" if whole_test else ""), ha="center", va="center", fontsize=12.5,
                    fontweight="bold", color=TEST if whole_test else INK)
            for c in range(2):
                y_line = 1.55 - c * 0.85
                xs = np.linspace(x0 + 0.35, x0 + 2.45, 5)
                ys = y_line + 0.28 - np.linspace(0, 0.56, 5)
                ax.plot(xs, ys, color="#c9c7c0", lw=1, zorder=1)
                ax.text(x0 + 0.05, y_line + 0.33, f"curve {c + 1}", fontsize=10, color=MUTED, va="bottom")
                for k in range(5):
                    is_test = (p, c, k) in test_sets[key]
                    ax.scatter(xs[k], ys[k], s=95, marker="s" if is_test else "o", color=TEST if is_test else TRAIN,
                               edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(9.5, 1.28, notes[key], fontsize=12.5, color=INK, va="center", ha="left", linespacing=1.35)
    handles = [plt.Line2D([], [], marker="o", ls="", color=TRAIN, markersize=11, label="measurement used for training"),
               plt.Line2D([], [], marker="s", ls="", color=TEST, markersize=11, label="measurement held out for testing")]
    fig.suptitle("How the three validation strategies split the same measurements", fontsize=16.5, fontweight="bold", y=0.998)
    fig.text(0.5, 0.958, "Illustrative example only: 3 papers, 2 S-N curves each, 5 measured fatigue lives per curve (not project data)",
             ha="center", fontsize=12, color=MUTED, style="italic")
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.012, "In the project the split is repeated five times (5-fold cross-validation), so every measurement, curve or paper is tested exactly once.\n"
             "Every tested measurement is a real experimental result: the model predicts its life from its inputs,\nand the prediction is compared with the measured life.",
             fontsize=11, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    return _save(fig, out_dir, "fig2_validation_strategies")


# ----------------------------------------------------------------------------------------------- Figure 3
def figure_3_all_models(values: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    x = np.arange(len(MODELS))
    width = 0.26
    for j, s in enumerate(SCHEMES):
        heights = [values["rmse"][(m, s)] for m in MODELS]
        bars = ax.bar(x + (j - 1) * width, heights, width=width * 0.94, color=SCHEME_COLORS[s], label=SCHEME_NAMES[s].replace("\n", " "), zorder=2)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=8.2, color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    mean_c = values["rmse"][("MeanPredictor", "C_article")]
    ax.axhline(mean_c, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.text(len(MODELS) - 0.5, mean_c + 0.012, f"average-life predictor: {mean_c:.2f}", ha="right", va="bottom", fontsize=8.3, color=MUTED)
    labels = [f"{MODEL_NAMES[m]}\nC/A = {values['inflation'][(m, 'C_article')][0]:.2f}" for m in MODELS]
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("RMSE of log10(fatigue life)  —  lower is better")
    ax.set_title("All five models under the three validation strategies", pad=10)
    ax.legend(ncol=3, loc="upper center", fontsize=9, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, "C/A = RMSE with whole articles held out divided by RMSE with random points (the 'inflation factor'). Primary experiment, seed 42.\n"
             "The y-axis starts at zero so that bar heights are proportional to the errors.", fontsize=7.8, color=MUTED, va="top")
    return _save(fig, out_dir, "fig3_all_models")


# ----------------------------------------------------------------------------------------------- Figure 4
def figure_4_actual_vs_predicted(data: dict, values: dict, out_dir: Path) -> list[Path]:
    pred = data["pred"][data["pred"]["model"] == "LightGBM"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    lims = (1.0, 10.0)
    for ax, s in zip(axes, ["A_random", "C_article"]):
        sub = pred[pred["scheme"] == s]
        hb = ax.hexbin(sub["y_true"], sub["y_pred"], gridsize=48, extent=(*lims, *lims), mincnt=1, cmap="Blues", bins="log", linewidths=0)
        ax.plot(lims, lims, color=INK, lw=1.1, label="perfect prediction")
        ax.plot(lims, (lims[0] + 0.3, lims[1] + 0.3), color=MUTED, lw=0.9, ls="--", label="within a factor of 2")
        ax.plot(lims, (lims[0] - 0.3, lims[1] - 0.3), color=MUTED, lw=0.9, ls="--")
        r = np.corrcoef(sub["y_true"], sub["y_pred"])[0, 1]
        ax.text(1.3, 9.6, f"RMSE {values['rmse'][('LightGBM', s)]:.3f}\nwithin a factor of 2: {100 * values['within_0.3'][('LightGBM', s)]:.0f}% of points\ncorrelation {r:.2f}",
                fontsize=8.4, va="top", color=INK, bbox=dict(fc="white", ec="#d6d4cf", boxstyle="round,pad=0.35"))
        ax.set_title({"A_random": "Random points (A)", "C_article": "Whole articles held out (C)"}[s])
        ax.set_xlabel("Measured fatigue life, log10(Nf)")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted fatigue life, log10(Nf)")
    axes[1].legend(loc="lower right", fontsize=8.3)
    cbar = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("measurements per hexagon (log scale)", fontsize=8.5)
    fig.suptitle("LightGBM: predicted against measured fatigue life (12,666 measurements)", fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.01, -0.02, "Each prediction was made by a model that had not seen that measurement (left) or that measurement's whole article (right).\n"
             "log10(Nf) = 5 means 100,000 cycles; a difference of 1 is a factor of 10 in life.", fontsize=7.8, color=MUTED, va="top")
    return _save(fig, out_dir, "fig4_actual_vs_predicted")


# ----------------------------------------------------------------------------------------------- Figure 5
def figure_5_article_offsets(data: dict, out_dir: Path) -> list[Path]:
    pred = data["pred"][data["pred"]["model"] == "LightGBM"]
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    bins = np.linspace(-3, 3, 49)
    for s in ["A_random", "C_article"]:
        sub = pred[pred["scheme"] == s]
        article_mean = (sub["y_pred"] - sub["y_true"]).groupby(sub["article_id"]).mean()
        label = {"A_random": "Random points (A)", "C_article": "Whole articles held out (C)"}[s]
        ax.hist(article_mean, bins=bins, histtype="step", lw=2.2, color=SCHEME_COLORS[s],
                label=f"{label}: spread {article_mean.std():.2f}")
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # headroom so the legend sits above the tallest bar
    ax.text(-0.08, ax.get_ylim()[1] * 0.97, "no shift", fontsize=8.5, color=MUTED, va="top", ha="right")
    ax.set_xlabel("Average error of one article's points, predicted minus measured log10(Nf)")
    ax.set_ylabel("Number of articles (of 440)")
    ax.set_title("On unseen articles the whole article is shifted up or down", pad=10)
    ax.legend(fontsize=8.6, loc="upper right", title="spread = standard deviation over the 440 articles", title_fontsize=8)
    fig.text(0.01, -0.03, "LightGBM, primary experiment. A value of +1 means the model over-predicted that article's lives by a factor of 10 on average.",
             fontsize=7.8, color=MUTED, va="top")
    return _save(fig, out_dir, "fig5_article_offsets")


def make_all(root, out_dir=None) -> list[Path]:
    """Write all five figures (PNG at 300 dpi and PDF) and return their paths."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "results" / "figures" / "beginner"
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    data = load(root)
    values = figure_values(data)
    paths = []
    paths += figure_1_central_finding(values, out_dir)
    paths += figure_2_validation_strategies(out_dir)
    paths += figure_3_all_models(values, out_dir)
    paths += figure_4_actual_vs_predicted(data, values, out_dir)
    paths += figure_5_article_offsets(data, out_dir)
    return paths


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    for path in make_all(project_root if len(sys.argv) < 2 else Path(sys.argv[1])):
        print(path)
