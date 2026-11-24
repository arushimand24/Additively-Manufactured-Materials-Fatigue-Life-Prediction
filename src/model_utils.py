"""Models, fold builder and cross-validation runner.

Design rules (see context files 06 and 07):
  * every fitted step (imputer, scaler, encoder, Basquin coefficients, model)
    is fitted on the training fold only;
  * hyperparameters are fixed in advance, there is no tuning;
  * grouped folds never share a group between train and test (asserted here
    and again in tests/test_leakage.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_utils import TARGET, assert_features_allowed

try:  # LightGBM is the main learner; scikit-learn's HistGradientBoosting is the documented fallback.
    from lightgbm import LGBMRegressor

    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover - only reached when LightGBM is not installed
    HAVE_LIGHTGBM = False

N_FOLDS = 5
DEFAULT_SEED = 42

# The column that defines the held-out unit for each validation scheme.
SCHEME_GROUP_COLUMN = {
    "A_random": "row_id",            # random points (optimistic reference)
    "B_curve": "curve_id",           # whole S-N curves held out
    "C_article": "article_id",       # whole source articles held out (main result)
    # Sensitivity schemes
    "C2_article_group": "article_group_id",   # articles that re-publish the same curves are held out together
    "D_author_cluster": "author_cluster_id",  # articles sharing an author are held out together
}

# Fixed hyperparameters (06_ml_methodology.md, section 6). Never tuned.
RIDGE_PARAMS = {"alpha": 1.0}
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": -1}
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,   # LightGBM applies row subsampling only when this is > 0
    "colsample_bytree": 0.8,
    "n_jobs": 4,
    "verbose": -1,
}
ONE_HOT_MIN_FREQUENCY = 20
BASQUIN_MIN_POINTS = 30
BASQUIN_MIN_CURVES = 3


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def make_folds(frame: pd.DataFrame, scheme: str, seed: int = DEFAULT_SEED, n_folds: int = N_FOLDS) -> pd.Series:
    """Assign every row to one of `n_folds` test folds for the given scheme.

    Steps (07_experimental_design_and_leakage.md, section 4):
      1. take the sorted unique group identifiers;
      2. shuffle them with numpy.random.default_rng(seed);
      3. cut the shuffled list into n_folds nearly equal parts;
      4. a row's fold is the part its group fell into.
    Grouped folds therefore hold equal numbers of *groups*, not of points.
    """
    group_col = SCHEME_GROUP_COLUMN[scheme]
    groups = np.array(sorted(frame[group_col].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    parts = np.array_split(shuffled, n_folds)
    fold_of_group = {group: fold for fold, part in enumerate(parts) for group in part}
    return frame[group_col].map(fold_of_group).astype(int).rename("fold")


def build_fold_table(frame: pd.DataFrame, schemes: list[str], seeds: list[int], n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Long table with one row per (row_id, scheme, seed) and its fold number."""
    tables = []
    for scheme in schemes:
        for seed in seeds:
            folds = make_folds(frame, scheme, seed, n_folds)
            tables.append(pd.DataFrame({"row_id": frame["row_id"].values, "scheme": scheme, "seed": seed, "fold": folds.values}))
    return pd.concat(tables, ignore_index=True)


def fold_composition(frame: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """Points, curves and articles in each test fold."""
    tmp = frame.assign(fold=folds.values)
    return tmp.groupby("fold").agg(points=("row_id", "size"), curves=("curve_id", "nunique"), articles=("article_id", "nunique")).reset_index()


def assert_group_disjoint(train: pd.DataFrame, test: pd.DataFrame, scheme: str) -> None:
    """Raise if a held-out group of this scheme also appears in the training rows."""
    group_col = SCHEME_GROUP_COLUMN[scheme]
    shared = set(train[group_col]) & set(test[group_col])
    if shared:
        raise AssertionError(f"Group leakage in scheme {scheme}: {len(shared)} shared values of {group_col}")


# --------------------------------------------------------------------------- #
# Basquin baseline
# --------------------------------------------------------------------------- #

class BasquinRegressor(BaseEstimator, RegressorMixin):
    """Basquin law log10(N) = A + B * log10(sigma_a), fitted by least squares.

    One line is fitted per alloy when the training data hold at least
    `min_points` points from at least `min_curves` curves for that alloy and the
    fitted slope B is negative (higher stress must give shorter life). Every
    other alloy, including alloys never seen in training, uses the pooled line
    fitted on all training rows.
    """

    needs_curve_ids = True

    def __init__(self, min_points: int = BASQUIN_MIN_POINTS, min_curves: int = BASQUIN_MIN_CURVES,
                 stress_col: str = "log_sa", alloy_col: str = "alloy"):
        self.min_points = min_points
        self.min_curves = min_curves
        self.stress_col = stress_col
        self.alloy_col = alloy_col

    @staticmethod
    def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Least-squares line y = a + b*x. Returns (a, b)."""
        slope, intercept = np.polyfit(x, y, deg=1)
        return float(intercept), float(slope)

    def fit(self, X: pd.DataFrame, y, curve_ids=None):
        x = X[self.stress_col].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        alloys = X[self.alloy_col].astype(str).to_numpy()
        curves = np.asarray(curve_ids) if curve_ids is not None else np.arange(len(X))

        self.pooled_ = self._fit_line(x, y)
        self.per_alloy_ = {}
        self.fit_log_ = []  # one record per alloy: why it got (or did not get) its own line
        for alloy in np.unique(alloys):
            mask = alloys == alloy
            n_points = int(mask.sum())
            n_curves = int(len(np.unique(curves[mask])))
            record = {"alloy": alloy, "train_points": n_points, "train_curves": n_curves}
            if n_points < self.min_points or n_curves < self.min_curves:
                record["decision"] = "pooled (too few points or curves)"
            elif np.ptp(x[mask]) == 0:
                record["decision"] = "pooled (single stress level)"
            else:
                intercept, slope = self._fit_line(x[mask], y[mask])
                record.update({"A": intercept, "B": slope})
                if slope < 0:
                    self.per_alloy_[alloy] = (intercept, slope)
                    record["decision"] = "per-alloy"
                else:
                    record["decision"] = "pooled (slope not negative)"
            self.fit_log_.append(record)
        return self

    def uses_pooled(self, X: pd.DataFrame) -> np.ndarray:
        """True for rows whose alloy has no per-alloy line (pooled fallback)."""
        alloys = X[self.alloy_col].astype(str).to_numpy()
        return np.array([alloy not in self.per_alloy_ for alloy in alloys])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.stress_col].to_numpy(dtype=float)
        alloys = X[self.alloy_col].astype(str).to_numpy()
        intercepts = np.empty(len(X))
        slopes = np.empty(len(X))
        for i, alloy in enumerate(alloys):
            intercepts[i], slopes[i] = self.per_alloy_.get(alloy, self.pooled_)
        return intercepts + slopes * x


# --------------------------------------------------------------------------- #
# Pipelines for the learned models
# --------------------------------------------------------------------------- #

def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Median imputation (+ optional scaling) for numerics, one-hot for categoricals.

    Categories seen fewer than ONE_HOT_MIN_FREQUENCY times in the *training fold*
    are merged into one "infrequent" column; categories unseen in training are
    mapped to that same column at prediction time, so rare and new alloys never
    break the model.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=ONE_HOT_MIN_FREQUENCY, sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", one_hot, categorical),
        ]
    )


def make_gbm(seed: int, **overrides):
    """The gradient-boosting learner with the fixed settings (LightGBM, or the fallback)."""
    if HAVE_LIGHTGBM:
        params = {**LGBM_PARAMS, "random_state": seed, **overrides}
        return LGBMRegressor(**params)
    # Fallback learner. A monotone constraint is passed on under scikit-learn's own parameter name,
    # so the constrained sensitivity model stays constrained without LightGBM.
    extra = {"monotonic_cst": overrides["monotone_constraints"]} if "monotone_constraints" in overrides else {}
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=20, random_state=seed, **extra)


def make_ridge(numeric, categorical, seed=DEFAULT_SEED) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric, categorical, scale=True)), ("model", Ridge(**RIDGE_PARAMS))])


def make_random_forest(numeric, categorical, seed=DEFAULT_SEED) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric, categorical, scale=False)),
                     ("model", RandomForestRegressor(random_state=seed, **RF_PARAMS))])


def make_lightgbm(numeric, categorical, seed=DEFAULT_SEED) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric, categorical, scale=False)), ("model", make_gbm(seed))])


class BasquinAnchoredResidual(BaseEstimator, RegressorMixin):
    """Physics-anchored hybrid: Basquin prediction plus a learned correction.

    fit:     1. fit the Basquin baseline on the training rows;
             2. compute training residuals r = y - y_Basquin;
             3. train the gradient-boosting pipeline on the same features to predict r.
    predict: y_Basquin + predicted r.
    """

    needs_curve_ids = True

    def __init__(self, numeric: list[str], categorical: list[str], seed: int = DEFAULT_SEED):
        self.numeric = numeric
        self.categorical = categorical
        self.seed = seed

    def fit(self, X: pd.DataFrame, y, curve_ids=None):
        y = np.asarray(y, dtype=float)
        self.basquin_ = BasquinRegressor().fit(X, y, curve_ids=curve_ids)
        residual = y - self.basquin_.predict(X)
        self.correction_ = make_lightgbm(self.numeric, self.categorical, self.seed).fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.basquin_.predict(X) + self.correction_.predict(X)

    def uses_pooled(self, X: pd.DataFrame) -> np.ndarray:
        return self.basquin_.uses_pooled(X)


def inner_group_folds(groups, n_folds: int = N_FOLDS, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, int]:
    """Inner folds over the given group labels with the documented fold rule (sorted unique groups,
    default_rng(seed) permutation, array_split). Used inside one outer training fold only."""
    groups = np.asarray(groups)
    unique = np.sort(np.unique(groups))
    k = int(min(n_folds, len(unique)))
    shuffled = np.random.default_rng(seed).permutation(unique)
    lookup = {g: i for i, part in enumerate(np.array_split(shuffled, k)) for g in part}
    return np.array([lookup[g] for g in groups]), k


class CrossFittedBasquinResidual(BaseEstimator, RegressorMixin):
    """Supplementary investigation A: the Basquin-anchored residual model with a cross-fitted anchor.

    The original hybrid (BasquinAnchoredResidual) trains its corrector on residuals of Basquin lines fitted on the
    same training rows (in-sample). Here, within one outer training fold:

    1. the outer training articles are split into `inner_folds` article-grouped inner folds (documented fold rule);
    2. for each inner fold, Basquin lines are fitted on the other inner folds and predict the held-out inner fold, so
       every outer training row gets an anchor prediction from lines fitted without its own article;
    3. the corrector (LightGBM, same settings) is trained on y minus these cross-fitted anchor predictions;
    4. the anchor used for prediction is the Basquin model fitted on all outer training rows.

    Fallback: in every inner and outer Basquin fit, an alloy without enough data (fewer than 30 points or 3 curves,
    a single stress level, or a slope that is not negative), or absent from that fit's rows, uses the pooled line of
    that fit's rows. Only outer training rows are ever used; the outer test articles never enter the fit.
    """

    needs_curve_ids = True
    needs_article_ids = True

    def __init__(self, numeric: list[str], categorical: list[str], seed: int = DEFAULT_SEED, inner_folds: int = N_FOLDS):
        self.numeric = numeric
        self.categorical = categorical
        self.seed = seed
        self.inner_folds = inner_folds

    def fit(self, X: pd.DataFrame, y, curve_ids=None, article_ids=None):
        if article_ids is None:
            raise ValueError("CrossFittedBasquinResidual needs the article id of every training row")
        y = np.asarray(y, dtype=float)
        curves = np.asarray(curve_ids)
        inner, k = inner_group_folds(article_ids, self.inner_folds, self.seed)
        anchor_oof = np.empty(len(y))
        for fold in range(k):
            held = inner == fold
            line = BasquinRegressor().fit(X[~held], y[~held], curve_ids=curves[~held])
            anchor_oof[held] = line.predict(X[held])
        self.inner_fold_ = inner
        self.anchor_oof_ = anchor_oof
        self.basquin_ = BasquinRegressor().fit(X, y, curve_ids=curves)
        self.correction_ = make_lightgbm(self.numeric, self.categorical, self.seed).fit(X, y - anchor_oof)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.basquin_.predict(X) + self.correction_.predict(X)

    def uses_pooled(self, X: pd.DataFrame) -> np.ndarray:
        return self.basquin_.uses_pooled(X)


class MonotoneGBM(BaseEstimator, RegressorMixin):
    """Sensitivity model: gradient boosting constrained so that predicted life never
    increases with stress amplitude (a real physical constraint).

    log_swt is left out of this model: it is a function of log_sa and R, so a
    constraint on log_sa "with everything else fixed" would be meaningless if a
    second stress column were free to move.
    """

    def __init__(self, numeric: list[str], categorical: list[str], seed: int = DEFAULT_SEED):
        self.numeric = numeric
        self.categorical = categorical
        self.seed = seed

    def fit(self, X: pd.DataFrame, y):
        self.numeric_ = [c for c in self.numeric if c != "log_swt"]
        self.prep_ = make_preprocessor(self.numeric_, self.categorical, scale=False)
        matrix = self.prep_.fit_transform(X[self.numeric_ + self.categorical])
        constraints = [0] * matrix.shape[1]
        constraints[self.numeric_.index("log_sa")] = -1  # numerics come first in the ColumnTransformer
        self.model_ = make_gbm(self.seed, monotone_constraints=constraints)
        self.model_.fit(matrix, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.prep_.transform(X[self.numeric_ + self.categorical]))


MODEL_ORDER = ["Basquin", "Ridge", "RandomForest", "LightGBM", "BasquinResidual"]
MODEL_LABELS = {
    "Basquin": "Basquin (per alloy)",
    "Ridge": "Ridge",
    "RandomForest": "Random Forest",
    "LightGBM": "LightGBM" if HAVE_LIGHTGBM else "HistGradientBoosting (LightGBM fallback)",
    "BasquinResidual": "Basquin-anchored residual",
    "LightGBM_monotone": "LightGBM, monotone in stress",
    "BasquinResidualCrossFitted": "Basquin-anchored residual, cross-fitted",
}


def make_model(name: str, numeric: list[str], categorical: list[str], seed: int = DEFAULT_SEED):
    """Build a fresh, unfitted model by name."""
    if name == "Basquin":
        return BasquinRegressor()
    if name == "Ridge":
        return make_ridge(numeric, categorical, seed)
    if name == "RandomForest":
        return make_random_forest(numeric, categorical, seed)
    if name == "LightGBM":
        return make_lightgbm(numeric, categorical, seed)
    if name == "BasquinResidual":
        return BasquinAnchoredResidual(numeric, categorical, seed)
    if name == "LightGBM_monotone":
        return MonotoneGBM(numeric, categorical, seed)
    if name == "BasquinResidualCrossFitted":
        return CrossFittedBasquinResidual(numeric, categorical, seed)
    raise ValueError(f"Unknown model name: {name}")


def fit_model(model, X: pd.DataFrame, y, curve_ids, article_ids=None):
    """Fit any of the project's models. Curve ids go only to the Basquin rule (to count curves per alloy), and
    article ids only to the cross-fitted hybrid (to form inner article-grouped folds); neither is a model feature."""
    kwargs = {}
    if getattr(model, "needs_curve_ids", False):
        kwargs["curve_ids"] = curve_ids
    if getattr(model, "needs_article_ids", False):
        kwargs["article_ids"] = article_ids
    return model.fit(X, y, **kwargs)


# --------------------------------------------------------------------------- #
# Cross-validation runner
# --------------------------------------------------------------------------- #

def run_cv(frame: pd.DataFrame, scheme: str, model_names: list[str], numeric: list[str], categorical: list[str],
           seed: int = DEFAULT_SEED, n_folds: int = N_FOLDS, target: str = TARGET, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold predictions for every model under one validation scheme.

    Returns
    -------
    predictions : one row per (row, model) with row_id, curve_id, article_id, alloy,
                  scheme, seed, fold, y_true, y_pred and, for Basquin-type models,
                  whether the pooled fallback line was used.
    basquin_log : per fold and alloy, the Basquin fitting decision (audit trail).
    """
    features = numeric + categorical
    assert_features_allowed(features)
    folds = make_folds(frame, scheme, seed, n_folds)
    X_all = frame[features]
    y_all = frame[target].to_numpy(dtype=float)

    prediction_parts = []
    basquin_logs = []
    for fold in range(n_folds):
        test_mask = (folds == fold).to_numpy()
        train, test = frame[~test_mask], frame[test_mask]
        assert len(test) > 0, "empty test fold"
        if scheme != "A_random":
            assert_group_disjoint(train, test, scheme)

        for name in model_names:
            model = make_model(name, numeric, categorical, seed)
            fit_model(model, X_all[~test_mask], y_all[~test_mask], curve_ids=train["curve_id"].to_numpy(),
                      article_ids=train["article_id"].to_numpy())
            y_pred = model.predict(X_all[test_mask])
            part = pd.DataFrame(
                {
                    "row_id": test["row_id"].to_numpy(),
                    "curve_id": test["curve_id"].to_numpy(),
                    "article_id": test["article_id"].to_numpy(),
                    "alloy": test["alloy"].to_numpy(),
                    "scheme": scheme,
                    "seed": seed,
                    "fold": fold,
                    "model": name,
                    "y_true": y_all[test_mask],
                    "y_pred": y_pred,
                }
            )
            if hasattr(model, "uses_pooled"):
                part["basquin_pooled_fallback"] = model.uses_pooled(X_all[test_mask])
            if name == "Basquin":
                log = pd.DataFrame(model.fit_log_)
                log.insert(0, "fold", fold)
                log.insert(0, "seed", seed)
                log.insert(0, "scheme", scheme)
                basquin_logs.append(log)
            prediction_parts.append(part)
        if verbose:
            print(f"  {scheme} seed {seed} fold {fold}: train {len(train):>6,} rows, test {len(test):>5,} rows")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    basquin_log = pd.concat(basquin_logs, ignore_index=True) if basquin_logs else pd.DataFrame()
    return predictions, basquin_log
