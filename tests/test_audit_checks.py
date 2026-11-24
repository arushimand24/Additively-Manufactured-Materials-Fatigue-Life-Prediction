"""Checks promoted from the technical audit (reports/independent_technical_audit.md) into the test suite.

They cover what the original leakage tests did not:
- every fold table (all schemes and seeds) follows the documented rule;
- every saved prediction file, not only the primary one, has one prediction per row and keeps groups together;
- the sensitivity runs use like-for-like folds;
- the cross-validation runner fits every model on training rows only (a spy records the rows it fits on);
- the near-duplicate search agrees with a brute-force search and treats ties symmetrically;
- the article groups and author clusters can be rebuilt, the article id is the DOI, the feature formulas reproduce
  from the raw workbook, and the provenance record matches the raw files and can be rebuilt offline.

Run from the project root:  python -m pytest tests -v
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import data_utils as du  # noqa: E402
import feature_utils as fu  # noqa: E402
import model_utils as mu  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDICTIONS = PROJECT_ROOT / "results" / "predictions"
RAW_XLSX = PROJECT_ROOT / "data" / "raw" / "FatigueData-AM2022.xlsx"
GROUP = mu.SCHEME_GROUP_COLUMN
PREDICTION_FILES = {
    "oof_predictions_primary.csv": "modelling", "oof_predictions_extra_seeds.csv": "modelling",
    "oof_predictions_group_sensitivity.csv": "modelling", "oof_predictions_monotone.csv": "modelling",
    "oof_predictions_tier1.csv": "modelling", "oof_predictions_uts_subset.csv": "uts", "oof_predictions_with_runouts.csv": "runouts",
    "oof_predictions_crossfitted_residual.csv": "modelling", "oof_predictions_ablation.csv": "modelling",
    "oof_predictions_regime_subsets.csv": "variant",
}
VARIANT_RULES = {"excluding tests at 1 kHz or above": lambda f: ~f["khz_test"],
                 "excluding displacement-controlled tests": lambda f: ~f["displacement_controlled"]}


def need(path):
    if not Path(path).exists():
        pytest.skip(f"{Path(path).name} not found; run the notebooks first")


@pytest.fixture(scope="module")
def frames():
    need(PROCESSED / "modelling_dataset.csv")
    modelling = pd.read_csv(PROCESSED / "modelling_dataset.csv")
    runouts = pd.read_csv(PROCESSED / "sn_cohort_with_runouts.csv")
    return {"modelling": modelling, "uts": modelling[modelling["log_uts"].notna()], "runouts": runouts}


def rule(frame, scheme, seed):
    """The documented fold rule, re-implemented: sorted groups, default_rng(seed) permutation, array_split into 5."""
    groups = np.sort(frame[GROUP[scheme]].unique())
    shuffled = np.random.default_rng(seed).permutation(groups)
    lookup = {g: k for k, part in enumerate(np.array_split(shuffled, mu.N_FOLDS)) for g in part}
    return frame[GROUP[scheme]].map(lookup).to_numpy()


# ----- fold tables ------------------------------------------------------------------------------------
def test_every_saved_fold_table_follows_the_documented_rule(frames):
    need(PROCESSED / "folds.csv")
    folds = pd.read_csv(PROCESSED / "folds.csv")
    data = frames["modelling"]
    assert folds.groupby(["scheme", "seed"]).ngroups >= 13
    for (scheme, seed), sub in folds.groupby(["scheme", "seed"]):
        got = sub.set_index("row_id")["fold"].reindex(data["row_id"]).to_numpy()
        assert (got == rule(data, scheme, seed)).all(), (scheme, seed)


# ----- every prediction file --------------------------------------------------------------------------
@pytest.mark.parametrize("file_name", sorted(PREDICTION_FILES))
def test_every_prediction_file_covers_each_row_once_and_keeps_groups_together(frames, file_name):
    need(PREDICTIONS / file_name)
    pred = pd.read_csv(PREDICTIONS / file_name)
    keys = [c for c in ["dataset_variant", "feature_set", "scheme", "seed", "model", "feature_tier"] if c in pred.columns]
    for key, sub in pred.groupby(keys):
        values = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        if PREDICTION_FILES[file_name] == "variant":          # the documented subset rule, applied to the modelling dataset
            regimes = fu.add_test_regime_columns(frames["modelling"])
            frame = regimes[VARIANT_RULES[values["dataset_variant"]](regimes)].reset_index(drop=True)
        else:
            frame = frames[PREDICTION_FILES[file_name]]
        truth = frame.set_index("row_id")["log_N"]
        assert sub["row_id"].is_unique and set(sub["row_id"]) == set(frame["row_id"]), values
        assert np.allclose(sub["y_true"].to_numpy(), truth.loc[sub["row_id"]].to_numpy()), values
        ordered = sub.set_index("row_id")["fold"].reindex(frame["row_id"]).to_numpy()
        assert (ordered == rule(frame, values["scheme"], values["seed"])).all(), values
        if values["scheme"] != "A_random":
            groups = frame.set_index("row_id")[GROUP[values["scheme"]]].loc[sub["row_id"]].to_numpy()
            assert pd.Series(sub["fold"].to_numpy()).groupby(groups).nunique().max() == 1, values


def test_sensitivity_runs_use_like_for_like_folds():
    need(PREDICTIONS / "oof_predictions_primary.csv")
    cols = ["row_id", "scheme", "model", "fold"]
    primary = pd.read_csv(PREDICTIONS / "oof_predictions_primary.csv", usecols=cols)
    base = primary.drop_duplicates(["scheme", "row_id"]).set_index(["scheme", "row_id"])["fold"]
    for name in ["oof_predictions_tier1.csv", "oof_predictions_monotone.csv"]:
        other = pd.read_csv(PREDICTIONS / name, usecols=cols).drop_duplicates(["scheme", "row_id"]).set_index(["scheme", "row_id"])["fold"]
        assert (other.reindex(base.index) == base).all(), name
    uts = pd.read_csv(PREDICTIONS / "oof_predictions_uts_subset.csv", usecols=cols + ["feature_tier"])
    t0 = uts[uts["feature_tier"] == "tier0"].drop_duplicates(["scheme", "row_id"]).set_index(["scheme", "row_id"])["fold"]
    t2 = uts[uts["feature_tier"] == "tier2"].drop_duplicates(["scheme", "row_id"]).set_index(["scheme", "row_id"])["fold"]
    assert (t2.reindex(t0.index) == t0).all()
    runouts = pd.read_csv(PREDICTIONS / "oof_predictions_with_runouts.csv", usecols=cols).drop_duplicates(["scheme", "row_id"]).set_index(["scheme", "row_id"])["fold"]
    article = base.loc["C_article"]
    assert (runouts.loc["C_article"].reindex(article.index) == article).all()


# ----- fitting on training rows only ------------------------------------------------------------------
def test_cross_validation_runner_fits_every_model_on_training_rows_only(frames, monkeypatch):
    data = frames["modelling"]
    small = data[data["article_id"].isin(np.sort(data["article_id"].unique())[:60])].copy()
    seen = []
    original = mu.fit_model

    def spy(model, X, y, curve_ids, **kwargs):
        seen.append((type(model).__name__, set(X.index)))
        return original(model, X, y, curve_ids, **kwargs)

    monkeypatch.setattr(mu, "fit_model", spy)
    for scheme in ["A_random", "C_article"]:
        seen.clear()
        mu.run_cv(small, scheme, ["Basquin", "Ridge", "BasquinResidual"], fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42, verbose=False)
        folds = mu.make_folds(small, scheme, 42)
        expected = [set(small.index[(folds != k).to_numpy()]) for k in range(mu.N_FOLDS) for _ in range(3)]
        assert len(seen) == len(expected)
        for (name, rows), train_rows in zip(seen, expected):
            assert rows == train_rows, (scheme, name)


# ----- near-duplicate curves --------------------------------------------------------------------------
def brute_force_pairs(frame, stress_tol=0.01, life_tol=0.02, min_shared=4, min_share=0.8):
    points = {c: np.log10(g[["stress_amplitude", "N"]].to_numpy(float)) for c, g in frame.groupby("dataset_id")}
    meta = frame.groupby("dataset_id").agg(article=("article_id", "first"), alloy=("material", "first"))
    found = set()
    for _, group in meta.groupby("alloy"):
        for a, b in combinations(group.index, 2):
            if group.at[a, "article"] == group.at[b, "article"]:
                continue
            pa, pb = points[a], points[b]

            def count(p, q):
                return int(((np.abs(p[:, None, 0] - q[None, :, 0]) <= stress_tol) & (np.abs(p[:, None, 1] - q[None, :, 1]) <= life_tol)).any(axis=1).sum())
            n = min(len(pa), len(pb))
            k = count(pa, pb) if len(pa) < len(pb) else count(pb, pa) if len(pb) < len(pa) else max(count(pa, pb), count(pb, pa))
            if k >= min_shared and k / n >= min_share:
                found.add((min(a, b), max(a, b)))
    return found


def test_near_duplicate_search_agrees_with_brute_force_on_the_documented_population():
    need(PROCESSED / "sn_cohort_with_runouts.csv")
    cohort = pd.read_csv(PROCESSED / "sn_cohort_with_runouts.csv").rename(columns={"curve_id": "dataset_id", "alloy": "material"})
    found = du.find_near_duplicate_curves(cohort)
    project = {(a, b) for a, b in zip(found["dataset_id_a"], found["dataset_id_b"])}
    saved = pd.read_csv(PROJECT_ROOT / "results" / "audit" / "near_duplicate_curves_in_cohort.csv", encoding="utf-8-sig")
    assert project == {(a, b) for a, b in zip(saved["dataset_id_a"], saved["dataset_id_b"])}
    subset = cohort[cohort["material"] != "Ti-6Al-4V"]            # the largest alloy is left out to keep the test fast
    expected = brute_force_pairs(subset)
    assert {p for p in project if p[0] in set(subset["dataset_id"])} == expected


def test_near_duplicate_ties_are_judged_symmetrically():
    # Two curves of four points each. All four points of curve P have a partner in curve Q, but only three of Q's
    # points have a partner in P (two of P's points sit next to the same point of Q). The counts are 4 one way
    # and 3 the other. The pair must be flagged, with 4 shared points, whichever curve has the lower id.
    p = {"stress_amplitude": [100.0, 120.0, 140.0, 140.5], "N": [1e4, 1e5, 1e6, 1e6]}
    q = {"stress_amplitude": [100.0, 120.0, 140.0, 160.0], "N": [1e4, 1e5, 1e6, 1e7]}
    for id_p, id_q in [(1, 2), (2, 1)]:
        frame = pd.concat([pd.DataFrame({**p, "dataset_id": id_p, "article_id": "x", "material": "M"}),
                           pd.DataFrame({**q, "dataset_id": id_q, "article_id": "y", "material": "M"})], ignore_index=True)
        result = du.find_near_duplicate_curves(frame)
        assert len(result) == 1 and result["shared_points"].iloc[0] == 4, (id_p, id_q)


def test_article_groups_are_the_union_of_near_duplicate_pairs(frames):
    data = frames["modelling"]
    saved = pd.read_csv(PROJECT_ROOT / "results" / "audit" / "near_duplicate_curves_in_cohort.csv", encoding="utf-8-sig")
    groups = du.connected_groups(sorted(frames["runouts"]["article_id"].unique()), list(zip(saved["article_id_a"], saved["article_id_b"])))
    assert (data["article_id"].map(groups) == data["article_group_id"]).all()
    assert data.groupby("article_id")["article_group_id"].nunique().max() == 1
    assert data.groupby("curve_id")["article_id"].nunique().max() == 1


# ----- raw-file checks (skipped when the workbook is absent) ------------------------------------------
@pytest.fixture(scope="module")
def raw(frames):
    need(RAW_XLSX)
    sn = pd.read_excel(RAW_XLSX, sheet_name="S-N")
    sn.columns = ["dataset_id", "N", "sa", "runout"]
    sn["row_id"] = np.arange(len(sn))
    par = pd.read_excel(RAW_XLSX, sheet_name="parameter", header=1).rename(columns={"dataset id": "dataset_id", "DOI": "doi"})
    return sn.merge(par[["dataset_id", "doi", "title", "authors"]], on="dataset_id")


def test_article_id_is_the_doi_and_no_article_is_split(frames, raw):
    m = frames["modelling"].merge(raw[["row_id", "doi", "title"]], on="row_id")
    assert (m["article_id"] == m["doi"].astype(str).str.strip().str.lower()).all()
    curves = raw.drop_duplicates("dataset_id").assign(t=lambda t: t["title"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    assert curves.groupby("t")["doi"].nunique().max() == 1 and curves.groupby("doi")["t"].nunique().max() == 1


def test_feature_formulas_reproduce_from_the_raw_workbook(frames, raw):
    m = frames["modelling"].merge(raw[["row_id", "N", "sa"]], on="row_id", suffixes=("", "_raw"))
    sigma_max = 2 * m["sa"] / (1 - m["R"])
    assert np.allclose(m["log_N"], np.log10(m["N_raw"]))
    assert np.allclose(m["log_sa"], np.log10(m["sa"]))
    assert np.allclose(m["log_swt"], np.log10(np.sqrt(sigma_max * m["sa"])))


def test_author_clusters_can_be_rebuilt_from_author_names(frames, raw):
    data = frames["modelling"]
    articles = sorted(frames["runouts"]["article_id"].unique())
    authors = raw.drop_duplicates("doi").assign(doi=lambda t: t["doi"].astype(str).str.strip().str.lower()).set_index("doi")["authors"]
    by_name = {}
    for article in articles:
        for name in str(authors.get(article, "")).split(";"):
            if name.strip():
                by_name.setdefault(name.strip().lower(), []).append(article)
    clusters = du.connected_groups(articles, [(v[0], w) for v in by_name.values() for w in v[1:]])
    rebuilt = data["article_id"].map(clusters)
    # same partition (labels may differ): each rebuilt cluster maps to exactly one saved cluster and vice versa
    assert pd.Series(data["author_cluster_id"].to_numpy()).groupby(rebuilt.to_numpy()).nunique().max() == 1
    assert pd.Series(rebuilt.to_numpy()).groupby(data["author_cluster_id"].to_numpy()).nunique().max() == 1


def test_provenance_record_matches_the_raw_files():
    need(du.PROVENANCE_PATH)
    record = json.loads(du.PROVENANCE_PATH.read_text(encoding="utf8"))
    for name, entry in record["files_downloaded"].items():
        need(du.RAW_DIR / name)
        assert du.sha256_of_file(du.RAW_DIR / name) == entry["sha256_on_disk"], name
        assert entry["md5_on_disk"] == entry["md5_reported_by_figshare"], name


def test_provenance_record_can_be_rebuilt_offline(tmp_path):
    payload = b"fatigue data"
    (tmp_path / "data.xlsx").write_bytes(payload)
    api = {"title": "T", "doi": "10.1/x.v2", "figshare_url": "https://example.invalid", "version": 2, "published_date": "2023-06-01",
           "license": {"name": "CC BY 4.0"}, "authors": [{"full_name": "A"}],
           "files": [{"name": "data.xlsx", "id": 1, "download_url": "u", "size": len(payload), "computed_md5": du.md5_of_file(tmp_path / "data.xlsx")},
                     {"name": "other.mat", "id": 2, "download_url": "v", "size": 1, "computed_md5": "0"}]}
    record = du.build_provenance_record(api, ["data.xlsx"], raw_dir=tmp_path)
    entry = record["files_downloaded"]["data.xlsx"]
    assert entry["md5_match"] and entry["sha256_on_disk"] == du.sha256_of_file(tmp_path / "data.xlsx")
    assert record["dataset_doi_versionless"] == "10.1/x" and record["files_available_but_not_downloaded"] == ["other.mat"]
    assert record["download_date_source"] == "modification time of the raw files on disk"
    # ensure_provenance_record never overwrites an existing record
    existing = tmp_path / "prov.json"
    existing.write_text(json.dumps({"keep": True}), encoding="utf8")
    assert du.ensure_provenance_record(["data.xlsx"], provenance_path=existing, api_record_path=tmp_path / "api.json") == {"keep": True}


# ----- supplementary investigation A: the cross-fitted hybrid ------------------------------------------
def test_cross_fitted_anchor_never_uses_a_rows_own_article(frames):
    data = frames["modelling"]
    train = data[data["article_id"].isin(np.sort(data["article_id"].unique())[:80])].reset_index(drop=True)
    model = mu.CrossFittedBasquinResidual(fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42)
    model.fit(train[fu.FEATURES_TIER0], train["log_N"].to_numpy(), curve_ids=train["curve_id"].to_numpy(),
              article_ids=train["article_id"].to_numpy())
    inner = pd.Series(model.inner_fold_)
    assert inner.groupby(train["article_id"]).nunique().max() == 1          # an article sits in one inner fold
    for fold in range(inner.max() + 1):
        held = (inner == fold).to_numpy()
        line = mu.BasquinRegressor().fit(train[~held], train.loc[~held, "log_N"].to_numpy(), curve_ids=train.loc[~held, "curve_id"].to_numpy())
        assert np.allclose(model.anchor_oof_[held], line.predict(train[held]))
        assert not set(train.loc[held, "article_id"]) & set(train.loc[~held, "article_id"])
    with pytest.raises(ValueError):
        mu.CrossFittedBasquinResidual(fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0).fit(train[fu.FEATURES_TIER0], train["log_N"], curve_ids=train["curve_id"])


def test_cross_fitted_hybrid_cannot_see_held_out_targets(frames):
    data = frames["modelling"]
    small = data[data["article_id"].isin(np.sort(data["article_id"].unique())[:60])].copy()
    base, _ = mu.run_cv(small, "C_article", ["BasquinResidualCrossFitted"], fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42, verbose=False)
    folds = mu.make_folds(small, "C_article", 42)
    altered = small.copy()
    altered.loc[(folds == 0).to_numpy(), "log_N"] += 3.0                  # change only the targets of held-out fold 0
    again, _ = mu.run_cv(altered, "C_article", ["BasquinResidualCrossFitted"], fu.NUMERIC_TIER0, fu.CATEGORICAL_TIER0, seed=42, verbose=False)
    fold0 = base["fold"] == 0
    assert np.allclose(base.loc[fold0, "y_pred"].to_numpy(), again.loc[fold0, "y_pred"].to_numpy())
