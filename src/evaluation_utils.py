"""Metrics, inflation factors, cluster bootstrap and plots.

All metrics are computed on out-of-fold predictions of log10(Nf), so every
number describes rows the model did not see during fitting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEME_LABELS = {
    "A_random": "A: random points",
    "B_curve": "B: curve held out",
    "C_article": "C: article held out",
    "C2_article_group": "C2: re-publication group held out",
    "D_author_cluster": "D: author cluster held out",
}
# Colour follows the scheme everywhere (fixed order, colour-blind-checked palette).
SCHEME_COLORS = {"A_random": "#2a78d6", "B_curve": "#eb6834", "C_article": "#1baf7a",
                 "C2_article_group": "#eda100", "D_author_cluster": "#6250d6"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#52514e"


def set_plot_style() -> None:
    """One quiet style for every figure: thin marks, light grid, no top/right frame."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b5b3ac",
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#e6e4df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "font.size": 10,
    })


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #

def regression_metrics(y_true, y_pred) -> dict:
    """RMSE and MAE of log10(Nf) first, then the secondary metrics.

    within_0.3 / within_0.5: share of predictions within +-0.3 / +-0.5 log10 units,
    i.e. within about a factor of 2 / 3 in life (10**0.3 = 2.0, 10**0.5 = 3.2).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "within_0.3": float(np.mean(np.abs(error) <= 0.3)),
        "within_0.5": float(np.mean(np.abs(error) <= 0.5)),
        "bias": float(np.mean(error)),
    }


def metrics_table(predictions: pd.DataFrame, by=("model", "scheme")) -> pd.DataFrame:
    """One row of metrics per group (default: per model and scheme)."""
    rows = []
    for keys, group in predictions.groupby(list(by), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row.update(regression_metrics(group["y_true"], group["y_pred"]))
        row["curves"] = int(group["curve_id"].nunique())
        row["articles"] = int(group["article_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def inflation_table(metrics: pd.DataFrame, reference: str = "A_random", value: str = "rmse") -> pd.DataFrame:
    """Inflation factor = metric under a grouped scheme / metric under the random scheme,
    for the same model. A value of 1.0 means no loss; 1.5 means the error is 50% larger."""
    wide = metrics.pivot(index="model", columns="scheme", values=value)
    out = wide.copy()
    for scheme in wide.columns:
        if scheme != reference:
            out[f"inflation_{scheme}"] = wide[scheme] / wide[reference]
    return out


def life_factor(log_error: float) -> float:
    """Convert an error in log10 units to a multiplicative factor in life."""
    return float(10 ** log_error)


# --------------------------------------------------------------------------- #
# Grouped views
# --------------------------------------------------------------------------- #

def per_article_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE of each article's points, per model and scheme."""
    tmp = predictions.assign(abs_error=(predictions["y_pred"] - predictions["y_true"]).abs())
    return (tmp.groupby(["model", "scheme", "article_id"])
            .agg(mae=("abs_error", "mean"), points=("abs_error", "size"))
            .reset_index())


def per_alloy_rmse(predictions: pd.DataFrame, min_articles: int = 1) -> pd.DataFrame:
    """RMSE per alloy with the number of points, curves and articles behind it."""
    tmp = predictions.assign(sq_error=(predictions["y_pred"] - predictions["y_true"]) ** 2)
    out = (tmp.groupby(["model", "scheme", "alloy"])
           .agg(rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))), points=("sq_error", "size"),
                curves=("curve_id", "nunique"), articles=("article_id", "nunique"))
           .reset_index())
    return out[out["articles"] >= min_articles]


def variance_components(frame: pd.DataFrame, value: str, article: str = "article_id", curve: str = "curve_id") -> dict:
    """Split the sum of squares of `value` into between-article, between-curve
    (within article) and within-curve parts. Descriptive only (no model is fitted).

    Shares answer: of the scatter that a stress-only law leaves unexplained, how much
    is a constant offset of the source article, of the curve, or point-to-point scatter?
    """
    v = frame[value].astype(float)
    grand = v.mean()
    article_mean = v.groupby(frame[article]).transform("mean")
    curve_mean = v.groupby(frame[curve]).transform("mean")
    ss_total = float(((v - grand) ** 2).sum())
    ss_article = float(((article_mean - grand) ** 2).sum())
    ss_curve = float(((curve_mean - article_mean) ** 2).sum())
    ss_within = float(((v - curve_mean) ** 2).sum())
    return {
        "ss_total": ss_total,
        "share_between_articles": ss_article / ss_total,
        "share_between_curves_within_article": ss_curve / ss_total,
        "share_within_curve": ss_within / ss_total,
    }


def error_decomposition(frame: pd.DataFrame, article: str = "article_id", curve: str = "curve_id",
                        error: str = "error") -> dict:
    """Split the squared out-of-fold error into an article part, a curve part and scatter.

    With e the error of one point (predicted minus actual log10 N), a_k the mean error of
    article k and c_j the mean error of curve j (curves are nested in articles):

        sum e^2 = sum_points a_k^2 + sum_points (c_j - a_k)^2 + sum_points (e - c_j)^2

    The identity is exact (the cross terms are zero for nested group means). The shares are
    point-weighted: an article with many points counts more. Assumptions and limits:

    - It is a descriptive partition of the error, not a model of its causes.
    - Group means are estimated from the same points, so pure point-to-point scatter also
      leaks into the article and curve parts. If the within-curve scatter were pure noise
      with variance s2 = SS_within / (n - curves), it would add about articles * s2 to the
      article part and (curves - articles) * s2 to the curve part. The "noise-corrected"
      shares subtract these expected amounts (a method-of-moments correction).
    - "article-weighted" averages each article's own split with equal weight per article.
    """
    e = frame[error].astype(float)
    article_mean = e.groupby(frame[article]).transform("mean")
    curve_mean = e.groupby(frame[curve]).transform("mean")
    ss_total = float((e ** 2).sum())
    ss_article = float((article_mean ** 2).sum())
    ss_curve = float(((curve_mean - article_mean) ** 2).sum())
    ss_within = float(((e - curve_mean) ** 2).sum())
    n, n_curves, n_articles = len(frame), frame[curve].nunique(), frame[article].nunique()
    s2 = ss_within / (n - n_curves)

    per_article = pd.DataFrame({"a": frame[article].to_numpy(), "e2": (e ** 2).to_numpy(),
                                "art": (article_mean ** 2).to_numpy(), "cur": ((curve_mean - article_mean) ** 2).to_numpy()})
    sums = per_article.groupby("a")[["e2", "art", "cur"]].sum()
    sums = sums[sums["e2"] > 0]
    return {
        "points": n, "curves": n_curves, "articles": n_articles,
        "article offset %": 100 * ss_article / ss_total,
        "curve offset within article %": 100 * ss_curve / ss_total,
        "scatter within curve %": 100 * ss_within / ss_total,
        "article offset %, noise-corrected": 100 * (ss_article - n_articles * s2) / ss_total,
        "curve offset %, noise-corrected": 100 * (ss_curve - (n_curves - n_articles) * s2) / ss_total,
        "scatter %, noise-corrected": 100 * n * s2 / ss_total,
        "article offset %, article-weighted": 100 * float((sums["art"] / sums["e2"]).mean()),
        "std of article mean errors": float(e.groupby(frame[article]).mean().std()),
    }


def absolute_error_summary(y_true, y_pred) -> dict:
    """Quantiles of the absolute error in log10 units and the life factors they correspond to.

    10**RMSE is a transformation of the RMSE, which squares the errors and is therefore pulled
    up by the largest ones. It is not the error of a typical prediction. The median absolute
    error, and the factor 10**median, describe a typical single prediction.
    """
    err = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    q50, q75, q90 = (float(np.quantile(err, q)) for q in (0.5, 0.75, 0.9))
    return {
        "rmse": rmse, "10^rmse": 10 ** rmse,
        "median abs error": q50, "factor at median": 10 ** q50,
        "75th percentile": q75, "factor at 75th percentile": 10 ** q75,
        "90th percentile": q90, "factor at 90th percentile": 10 ** q90,
        "share off by more than a factor of 10": float(np.mean(err > 1.0)),
    }


# --------------------------------------------------------------------------- #
# Cluster bootstrap over articles
# --------------------------------------------------------------------------- #

def cluster_bootstrap_rmse(predictions: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                           reference: str = "A_random") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap RMSE and inflation factors by resampling whole articles.

    Points inside an article are not independent, so resampling points would make
    the intervals far too narrow. Each resample draws articles with replacement and
    keeps all of their points. The same resampled articles are used for every model
    and scheme, so differences between models are paired.

    What the intervals cover. The function resamples the saved OUT-OF-FOLD predictions;
    nothing is refitted. The intervals therefore describe how much a metric would change
    if a different set of articles had been evaluated, conditional on the fitted models,
    the fold assignment and the fixed hyperparameters. They do not include the variation
    from refitting on a different training sample, from the fold assignment (shown
    separately by repeating the procedure with other seeds) or from model selection.

    Returns (samples, summary). `samples` has one column per (model, scheme) and
    n_boot rows of RMSE values. `summary` gives the point estimate and the 2.5 and
    97.5 percentiles for each RMSE and each inflation factor.
    """
    tmp = predictions.assign(sq_error=(predictions["y_pred"] - predictions["y_true"]) ** 2)
    per_article = tmp.groupby(["model", "scheme", "article_id"]).agg(sse=("sq_error", "sum"), n=("sq_error", "size"))
    sse = per_article["sse"].unstack(["model", "scheme"]).fillna(0.0)
    count = per_article["n"].unstack(["model", "scheme"]).fillna(0.0)
    articles = sse.index.to_numpy()
    sse_matrix, count_matrix = sse.to_numpy(), count.to_numpy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(articles), size=(n_boot, len(articles)))
    boot_rmse = np.empty((n_boot, sse_matrix.shape[1]))
    for b in range(n_boot):
        idx = draws[b]
        boot_rmse[b] = np.sqrt(sse_matrix[idx].sum(axis=0) / count_matrix[idx].sum(axis=0))
    samples = pd.DataFrame(boot_rmse, columns=sse.columns)

    point = np.sqrt(sse_matrix.sum(axis=0) / count_matrix.sum(axis=0))
    rows = []
    for j, (model, scheme) in enumerate(sse.columns):
        rows.append({"model": model, "scheme": scheme, "quantity": "rmse", "estimate": point[j],
                     "ci_low": np.percentile(boot_rmse[:, j], 2.5), "ci_high": np.percentile(boot_rmse[:, j], 97.5)})
    for model in sse.columns.get_level_values("model").unique():
        if (model, reference) not in sse.columns:
            continue
        j_ref = list(sse.columns).index((model, reference))
        for scheme in sse.columns.get_level_values("scheme").unique():
            if scheme == reference or (model, scheme) not in sse.columns:
                continue
            j = list(sse.columns).index((model, scheme))
            ratio = boot_rmse[:, j] / boot_rmse[:, j_ref]
            rows.append({"model": model, "scheme": scheme, "quantity": "inflation_vs_random",
                         "estimate": point[j] / point[j_ref],
                         "ci_low": np.percentile(ratio, 2.5), "ci_high": np.percentile(ratio, 97.5)})
    return samples, pd.DataFrame(rows)


def bootstrap_difference(samples: pd.DataFrame, first: tuple, second: tuple) -> dict:
    """Paired bootstrap summary of RMSE(first) - RMSE(second), each a (model, scheme) pair.

    `bootstrap_mean_difference` is the mean over the resamples, not the difference of the point estimates;
    report the point difference from the full data and use this function for the interval.
    """
    diff = samples[first] - samples[second]
    return {
        "bootstrap_mean_difference": float(diff.mean()),
        "ci_low": float(np.percentile(diff, 2.5)),
        "ci_high": float(np.percentile(diff, 97.5)),
        "share_of_resamples_first_is_lower": float((diff < 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_predicted_vs_actual(predictions: pd.DataFrame, model: str, schemes: list[str], title_prefix: str = "") -> plt.Figure:
    """Predicted against actual log10(Nf), one panel per scheme, with factor-of-2 and factor-of-3 bands."""
    fig, axes = plt.subplots(1, len(schemes), figsize=(5.2 * len(schemes), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, scheme in zip(axes, schemes):
        sub = predictions[(predictions["model"] == model) & (predictions["scheme"] == scheme)]
        ax.scatter(sub["y_true"], sub["y_pred"], s=5, alpha=0.25, color=SCHEME_COLORS.get(scheme, "C0"), rasterized=True)
        lims = [min(sub["y_true"].min(), sub["y_pred"].min()) - 0.2, max(sub["y_true"].max(), sub["y_pred"].max()) + 0.2]
        ax.plot(lims, lims, color="black", lw=1)
        for band, style in ((0.3, "--"), (0.5, ":")):
            ax.plot(lims, [lims[0] + band, lims[1] + band], color="black", lw=0.8, ls=style)
            ax.plot(lims, [lims[0] - band, lims[1] - band], color="black", lw=0.8, ls=style)
        m = regression_metrics(sub["y_true"], sub["y_pred"])
        ax.set_title(f"{SCHEME_LABELS.get(scheme, scheme)}\nRMSE {m['rmse']:.3f}, MAE {m['mae']:.3f}, within ±0.3: {100 * m['within_0.3']:.0f}%")
        ax.set_xlabel("actual log10(Nf)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
    axes[0].set_ylabel("out-of-fold predicted log10(Nf)")
    fig.suptitle(f"{title_prefix}{model}: predicted vs actual (dashed ±0.3 ≈ factor 2, dotted ±0.5 ≈ factor 3)", y=1.02)
    fig.tight_layout()
    return fig


def plot_rmse_by_model_and_scheme(metrics: pd.DataFrame, model_order: list[str], schemes: list[str],
                                  model_labels: dict | None = None, ci: pd.DataFrame | None = None) -> plt.Figure:
    """Grouped bar chart of RMSE; optional 95% bootstrap intervals as error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(schemes)
    x = np.arange(len(model_order))
    for k, scheme in enumerate(schemes):
        values, lows, highs = [], [], []
        for model in model_order:
            row = metrics[(metrics["model"] == model) & (metrics["scheme"] == scheme)]
            value = float(row["rmse"].iloc[0])
            values.append(value)
            if ci is not None:
                c = ci[(ci["model"] == model) & (ci["scheme"] == scheme) & (ci["quantity"] == "rmse")]
                lows.append(value - float(c["ci_low"].iloc[0]))
                highs.append(float(c["ci_high"].iloc[0]) - value)
        positions = x + (k - (len(schemes) - 1) / 2) * width
        ax.bar(positions, values, width=width * 0.95, color=SCHEME_COLORS.get(scheme, None), label=SCHEME_LABELS.get(scheme, scheme),
               yerr=[lows, highs] if ci is not None else None, capsize=3, error_kw={"lw": 1})
        for pos, value in zip(positions, values):
            # Value labels sit inside the bar base so they never collide with the interval whiskers.
            ax.text(pos, 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([(model_labels or {}).get(m, m) for m in model_order])
    ax.set_ylabel("RMSE of log10(Nf), out-of-fold")
    ax.set_title("Error by model and validation scheme" + (" (bars: 95% article-bootstrap interval)" if ci is not None else ""))
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path, dpi: int = 150) -> None:
    """Save a figure as PNG, creating the folder if needed."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Small reporting helpers
# --------------------------------------------------------------------------- #

def df_to_markdown(frame: pd.DataFrame, float_format: str = "{:.3f}", index: bool = True) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""
    table = frame.reset_index() if index else frame
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else float_format.format(value)
        return str(value)
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def rmse_pivot(predictions: pd.DataFrame, models: list[str], schemes: list[str], reference: str = "A_random") -> pd.DataFrame:
    """RMSE per model (rows) and scheme (columns) plus inflation factors against the reference scheme."""
    table = metrics_table(predictions).pivot(index="model", columns="scheme", values="rmse").loc[models, schemes]
    for scheme in schemes:
        if scheme != reference and reference in schemes:
            table[f"inflation {scheme}/{reference}"] = table[scheme] / table[reference]
    return table


def mean_baseline_predictions(frame: pd.DataFrame, folds: pd.Series, target: str = "log_N") -> np.ndarray:
    """Out-of-fold predictions of the simplest possible model: the mean life of the training fold."""
    prediction = np.empty(len(frame))
    y = frame[target].to_numpy(dtype=float)
    fold_values = folds.to_numpy()
    for fold in np.unique(fold_values):
        test = fold_values == fold
        prediction[test] = y[~test].mean()
    return prediction
