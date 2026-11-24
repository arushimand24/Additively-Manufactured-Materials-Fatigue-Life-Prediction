"""Leakage and correctness tests for the FatigueData-AM2022 generalisation study.

Run from the project root with:  python -m pytest tests -v

Tests 1-7 are the seven automated tests required by
`07_experimental_design_and_leakage.md`, section 10. The remaining tests check
the Basquin rules, that preprocessing only learns from training rows, the stress
definitions, the error decomposition and the temperature labels.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import data_utils as du  # noqa: E402
import evaluation_utils as eu  # noqa: E402
import feature_utils as fu  # noqa: E402
import model_utils as mu  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "modelling_dataset.csv"
FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "folds.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions" / "oof_predictions_primary.csv"

ALL_FEATURE_TIERS = [fu.FEATURES_TIER0, fu.FEATURES_TIER1, fu.FEATURES_TIER2]


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        pytest.skip("modelling_dataset.csv not found; run notebook 02 first")
    return pd.read_csv(DATASET_PATH)


# ----- Required test 1: curve-held-out folds share no curve ---------------------------------
def test_01_no_curve_in_both_train_and_test(dataset):
    folds = mu.make_folds(dataset, "B_curve", seed=42)
    for fold in range(mu.N_FOLDS):
        test_curves = set(dataset.loc[folds == fold, "curve_id"])
        train_curves = set(dataset.loc[folds != fold, "curve_id"])
        assert not (test_curves & train_curves), f"curve leakage in fold {fold}"


# ----- Required test 2: article-held-out folds share no article -----------------------------
def test_02_no_article_in_both_train_and_test(dataset):
    folds = mu.make_folds(dataset, "C_article", seed=42)
    for fold in range(mu.N_FOLDS):
        test_articles = set(dataset.loc[folds == fold, "article_id"])
        train_articles = set(dataset.loc[folds != fold, "article_id"])
        assert not (test_articles & train_articles), f"article leakage in fold {fold}"


def test_02b_article_folds_also_separate_curves(dataset):
    """A curve belongs to one article, so article-held-out folds must also be curve-disjoint."""
    assert dataset.groupby("curve_id")["article_id"].nunique().max() == 1
    folds = mu.make_folds(dataset, "C_article", seed=42)
    assert dataset.assign(fold=folds).groupby("curve_id")["fold"].nunique().max() == 1


def test_02c_sensitivity_groups_are_disjoint(dataset):
    for scheme in ("C2_article_group", "D_author_cluster"):
        column = mu.SCHEME_GROUP_COLUMN[scheme]
        folds = mu.make_folds(dataset, scheme, seed=42)
        assert dataset.assign(fold=folds).groupby(column)["fold"].nunique().max() == 1
        # Each article sits inside exactly one group, so these schemes are also article-disjoint.
        assert dataset.assign(fold=folds).groupby("article_id")["fold"].nunique().max() == 1


# ----- Required test 3: the run-out flag is not a feature ------------------------------------
def test_03_runout_flag_not_in_features():
    for features in ALL_FEATURE_TIERS:
        assert "runout" not in features and "is_runout" not in features


# ----- Required test 4: the target is not a feature ------------------------------------------
def test_04_target_not_in_features():
    for features in ALL_FEATURE_TIERS:
        assert fu.TARGET not in features and "N" not in features and "log_N" not in features


# ----- Required test 5: no identifier is a feature -------------------------------------------
def test_05_no_identifier_in_features():
    identifiers = {"row_id", "curve_id", "dataset_id", "article_id", "article_group_id", "author_cluster_id",
                   "doi", "title", "authors", "institution", "country", "year", "journal", "rating_score", "extraction_method"}
    for features in ALL_FEATURE_TIERS:
        assert not (set(features) & identifiers)
        assert not (set(features) & set(fu.FORBIDDEN))


def test_05b_forbidden_feature_is_rejected():
    with pytest.raises(ValueError):
        fu.assert_features_allowed(fu.FEATURES_TIER0 + ["article_id"])
    with pytest.raises(ValueError):
        fu.assert_features_allowed(fu.FEATURES_TIER0 + ["runout"])


# ----- Required test 6: the same seed gives the same folds -----------------------------------
def test_06_same_seed_same_folds(dataset):
    for scheme in ("A_random", "B_curve", "C_article"):
        first = mu.make_folds(dataset, scheme, seed=42)
        second = mu.make_folds(dataset.sample(frac=1.0, random_state=1), scheme, seed=42).reindex(dataset.index)
        assert (first == second).all(), f"fold assignment of {scheme} is not reproducible"
    assert not (mu.make_folds(dataset, "C_article", seed=42) == mu.make_folds(dataset, "C_article", seed=43)).all()


def test_06b_saved_fold_table_matches_the_rule(dataset):
    if not FOLDS_PATH.exists():
        pytest.skip("folds.csv not found; run notebook 03 first")
    saved = pd.read_csv(FOLDS_PATH)
    for scheme in ("A_random", "B_curve", "C_article"):
        table = saved[(saved["scheme"] == scheme) & (saved["seed"] == 42)].set_index("row_id")["fold"]
        rebuilt = pd.Series(mu.make_folds(dataset, scheme, seed=42).to_numpy(), index=dataset["row_id"])
        assert (table.reindex(rebuilt.index) == rebuilt).all()


# ----- Required test 7: exactly one out-of-fold prediction per row, scheme and model ---------
def test_07_one_out_of_fold_prediction_per_row_scheme_model(dataset):
    if not PREDICTIONS_PATH.exists():
        pytest.skip("out-of-fold predictions not found; run notebook 03 first")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    counts = predictions.groupby(["scheme", "model", "row_id"]).size()
    assert (counts == 1).all(), "a row has more than one prediction for some scheme and model"
    for (scheme, model), group in predictions.groupby(["scheme", "model"]):
        assert set(group["row_id"]) == set(dataset["row_id"]), f"rows missing for {scheme} / {model}"
    assert set(predictions["scheme"]) == {"A_random", "B_curve", "C_article"}
    assert set(predictions["model"]) == set(mu.MODEL_ORDER)
    assert np.isfinite(predictions["y_pred"]).all()


def test_07b_saved_predictions_respect_group_separation():
    """Re-derive train/test membership from the saved predictions and check the groups again."""
    if not PREDICTIONS_PATH.exists():
        pytest.skip("out-of-fold predictions not found; run notebook 03 first")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    one_model = predictions[predictions["model"] == mu.MODEL_ORDER[0]]
    for scheme, column in (("B_curve", "curve_id"), ("C_article", "article_id")):
        sub = one_model[one_model["scheme"] == scheme]
        assert sub.groupby(column)["fold"].nunique().max() == 1


# ----- Basquin rules ---------------------------------------------------------------------------
def _synthetic_alloy(alloy, intercept, slope, n_curves, points_per_curve, rng):
    log_sa = rng.uniform(2.0, 2.8, size=n_curves * points_per_curve)
    return pd.DataFrame({
        "alloy": alloy,
        "log_sa": log_sa,
        "curve_id": np.repeat(np.arange(n_curves), points_per_curve),
        "log_N": intercept + slope * log_sa,
    })


def test_basquin_recovers_known_coefficients_and_uses_fallbacks():
    rng = np.random.default_rng(0)
    big = _synthetic_alloy("big", 20.0, -6.0, n_curves=5, points_per_curve=10, rng=rng)             # per-alloy fit
    few_points = _synthetic_alloy("few_points", 15.0, -4.0, n_curves=3, points_per_curve=5, rng=rng)  # < 30 points
    few_curves = _synthetic_alloy("few_curves", 15.0, -4.0, n_curves=2, points_per_curve=20, rng=rng)  # < 3 curves
    rising = _synthetic_alloy("rising", 1.0, 2.0, n_curves=4, points_per_curve=10, rng=rng)          # positive slope
    train = pd.concat([big, few_points, few_curves, rising], ignore_index=True)
    train["curve_id"] = train["alloy"] + "_" + train["curve_id"].astype(str)

    model = mu.BasquinRegressor().fit(train[["log_sa", "alloy"]], train["log_N"], curve_ids=train["curve_id"])
    assert set(model.per_alloy_) == {"big"}
    intercept, slope = model.per_alloy_["big"]
    assert abs(intercept - 20.0) < 1e-6 and abs(slope + 6.0) < 1e-6

    decisions = {record["alloy"]: record["decision"] for record in model.fit_log_}
    assert decisions["few_points"].startswith("pooled") and decisions["few_curves"].startswith("pooled")
    assert decisions["rising"] == "pooled (slope not negative)"

    unseen = pd.DataFrame({"log_sa": [2.5], "alloy": ["never_seen"]})
    assert model.uses_pooled(unseen).all()
    assert np.isclose(model.predict(unseen)[0], model.pooled_[0] + model.pooled_[1] * 2.5)


def test_heldout_targets_cannot_influence_their_own_predictions(dataset):
    """The strongest leakage test: change the lives of one held-out fold and predict it again.

    If any step of any model (imputer, encoder, scaler, Basquin fit, learner) saw the
    held-out rows during fitting, the predictions for that fold would change. They must
    be identical. As a sanity check, the predictions of another fold, whose training set
    contains the altered rows, must change.
    """
    articles = sorted(dataset["article_id"].unique())[:90]            # a subset keeps the test fast
    sample = dataset[dataset["article_id"].isin(articles)].reset_index(drop=True)
    folds = mu.make_folds(sample, "C_article", seed=42).to_numpy()

    altered = sample.copy()
    in_fold_0 = folds == 0
    altered.loc[in_fold_0, fu.TARGET] = altered.loc[in_fold_0, fu.TARGET].to_numpy()[::-1] + 3.0

    original, _ = mu.run_cv(sample, "C_article", mu.MODEL_ORDER, fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42, verbose=False)
    changed, _ = mu.run_cv(altered, "C_article", mu.MODEL_ORDER, fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42, verbose=False)
    merged = original.merge(changed, on=["row_id", "model", "fold"], suffixes=("_original", "_changed"))
    assert len(merged) == len(original)

    held_out = merged[merged["fold"] == 0]
    assert np.allclose(held_out["y_pred_original"], held_out["y_pred_changed"]), "held-out targets leaked into their own predictions"
    for model in mu.MODEL_ORDER:
        other = merged[(merged["fold"] == 1) & (merged["model"] == model)]
        assert not np.allclose(other["y_pred_original"], other["y_pred_changed"]), f"{model}: altering training targets had no effect"


# ----- Preprocessing is learned from training rows only ----------------------------------------
def test_preprocessing_learns_categories_from_training_rows_only(dataset):
    folds = mu.make_folds(dataset, "C_article", seed=42)
    train, test = dataset[folds != 0], dataset[folds == 0]
    prep = mu.make_preprocessor(fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, scale=True).fit(train[fu.FEATURES_TIER0])
    encoder = prep.named_transformers_["cat"]
    alloy_position = fu.CATEGORICAL_TIER0.index("alloy")
    learned_alloys = set(encoder.categories_[alloy_position])
    assert learned_alloys == set(train["alloy"]), "encoder categories must come from the training fold only"
    # Alloys that exist only in the test fold must still be transformed without error.
    transformed = prep.transform(test[fu.FEATURES_TIER0])
    assert transformed.shape[0] == len(test) and np.isfinite(transformed).all()
    # The scaler's centre is the training mean, not the full-data mean.
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    assert np.isclose(scaler.mean_[0], train["log_sa"].mean())


def test_residual_model_is_basquin_plus_correction(dataset):
    sample = dataset.sample(n=1500, random_state=0)
    model = mu.BasquinAnchoredResidual(fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42)
    model.fit(sample[fu.FEATURES_TIER0], sample[fu.TARGET], curve_ids=sample["curve_id"])
    combined = model.predict(sample[fu.FEATURES_TIER0])
    parts = model.basquin_.predict(sample[fu.FEATURES_TIER0]) + model.correction_.predict(sample[fu.FEATURES_TIER0])
    assert np.allclose(combined, parts)


# ----- Stress definitions: R = sigma_min / sigma_max and the stress-based SWT form ----------
def test_stress_features_follow_the_dataset_conventions():
    sigma_max = np.array([500.0, 300.0, 200.0, 120.0])
    r = np.array([0.1, -1.0, 0.5, -3.0])
    sigma_min = r * sigma_max
    sa = (sigma_max - sigma_min) / 2          # equals (1 - R) / 2 * sigma_max, as in the dataset paper
    out = fu.add_stress_features(pd.DataFrame({"stress_amplitude": sa, "R": r}))
    assert np.allclose(out["sigma_max"], sigma_max)
    assert np.allclose(out["sigma_mean"], (sigma_max + sigma_min) / 2)
    swt = 10 ** out["log_swt"].to_numpy()
    assert np.allclose(swt, np.sqrt(sigma_max * sa))      # sqrt(sigma_max * sigma_a)
    assert np.isclose(swt[1], sa[1])                       # fully reversed: equals the amplitude
    assert swt[0] > sa[0] and swt[2] > sa[2]               # tensile mean stress: larger than sa
    assert swt[3] < sa[3]                                  # compression-dominated: smaller than sa


# ----- Error decomposition: exact partition, and it finds a planted article offset ----------
def test_error_decomposition_is_exact_and_recovers_article_offsets():
    rng = np.random.default_rng(0)
    rows = []
    for article in range(60):
        offset = rng.normal(0, 1.0)
        for curve in range(3):
            shift = rng.normal(0, 0.3)
            for _ in range(10):
                rows.append({"article_id": article, "curve_id": f"{article}-{curve}", "error": offset + shift + rng.normal(0, 0.5)})
    frame = pd.DataFrame(rows)
    result = eu.error_decomposition(frame)
    raw = result["article offset %"] + result["curve offset within article %"] + result["scatter within curve %"]
    corrected = result["article offset %, noise-corrected"] + result["curve offset %, noise-corrected"] + result["scatter %, noise-corrected"]
    assert np.isclose(raw, 100.0) and np.isclose(corrected, 100.0)
    # planted variances 1.0 : 0.09 : 0.25, so the article part dominates and the noise correction lowers it
    assert result["article offset %"] > 60
    assert result["article offset %, noise-corrected"] < result["article offset %"]
    assert result["scatter %, noise-corrected"] > result["scatter within curve %"]


# ----- Temperature labels: 25 degC is only taken as reported when the article shows it reports temperature
def test_temperature_evidence_labels():
    frame = pd.DataFrame({"article_id": ["a", "a", "b", "c"], "test_temperature_c": [25, 650, 25, 20]})
    labels = du.temperature_evidence(frame).tolist()
    assert labels == ["25, article reports other temps", "explicit, not the default", "25, default or reported", "explicit, not the default"]
    assert "temperature_evidence" in fu.FORBIDDEN
