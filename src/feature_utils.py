"""Feature engineering rules for the FatigueData-AM2022 generalisation study.

Everything here is a fixed, documented rule that uses only information known
*before* a fatigue test. Nothing is fitted to data in this module, so the same
rules can be applied to the whole table before splitting without leakage.

The single source of truth for what a model may see is `FEATURES_TIER0`
(and the optional tiers). `FORBIDDEN` lists columns that must never be model
inputs; `assert_features_allowed` enforces it and is called by the CV runner
and by the tests.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Feature lists
# --------------------------------------------------------------------------- #

TARGET = "log_N"

NUMERIC_TIER0 = ["log_sa", "R", "log_swt"]
CATEGORICAL_TIER0 = ["alloy", "am_process", "surface_class", "heat_treatment_class", "test_type_class"]
FEATURES_TIER0 = NUMERIC_TIER0 + CATEGORICAL_TIER0

# Tier 1: added only in a sensitivity run, after coverage was checked in notebook 02.
NUMERIC_TIER1 = NUMERIC_TIER0 + ["kt", "log_frequency"]
CATEGORICAL_TIER1 = CATEGORICAL_TIER0 + ["build_direction_class"]
FEATURES_TIER1 = NUMERIC_TIER1 + CATEGORICAL_TIER1

# Tier 2: static-strength features, only on the subset where UTS exists.
NUMERIC_TIER2 = NUMERIC_TIER0 + ["sa_over_uts", "log_uts"]
CATEGORICAL_TIER2 = CATEGORICAL_TIER0
FEATURES_TIER2 = NUMERIC_TIER2 + CATEGORICAL_TIER2

# Identifier columns: used for grouping, joining and auditing only.
ID_COLUMNS = ["row_id", "curve_id", "article_id", "article_group_id", "author_cluster_id"]

# Columns that must never be model inputs (rule 2 and 3 of the project index).
FORBIDDEN = [
    # the target and anything computed from it
    "N", "log_N", "runout", "is_runout",
    # identifiers
    "row_id", "curve_id", "dataset_id", "article_id", "article_group_id", "author_cluster_id", "in_near_duplicate_pair",
    "doi", "title", "authors", "institution", "country", "funding", "year", "journal",
    # audit fields that can act as a proxy for the source
    "rating_score", "extraction_method", "temperature_evidence", "test_machine",
]


def assert_features_allowed(features: list[str]) -> None:
    """Raise if any forbidden column is in the feature list."""
    overlap = sorted(set(features) & set(FORBIDDEN))
    if overlap:
        raise ValueError(f"Forbidden columns used as features: {overlap}")


# Check the built-in tiers once at import time.
for _tier in (FEATURES_TIER0, FEATURES_TIER1, FEATURES_TIER2):
    assert_features_allowed(_tier)


# --------------------------------------------------------------------------- #
# Stress features
# --------------------------------------------------------------------------- #

def add_stress_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add log_sa, log_swt and the plotting helpers sigma_max and sigma_mean.

    Conventions (as in the dataset paper, Zhang & Xu 2023, Methods):
      - sa is the nominal stress amplitude in MPa, (sigma_max - sigma_min) / 2.
      - R = sigma_min / sigma_max (R = -1 fully reversed, R = 0.1 tension-tension).
        The paper converts reported maxima to amplitudes with sa = (1 - R) / 2 * sigma_max,
        so sigma_max below recovers the reported maximum for those curves.

    Formulas, all in MPa and defined for R < 1 (guaranteed by the filter funnel):
      sigma_max  = 2 * sa / (1 - R)
      sigma_mean = sa * (1 + R) / (1 - R)
      sigma_SWT  = sqrt(sigma_max * sa) = sa * sqrt(2 / (1 - R))
      log_swt    = log10(sigma_SWT)

    What sigma_SWT is. Smith, Watson and Topper (1970) defined their parameter from the
    strain amplitude, sqrt(sigma_max * eps_a * E). With linear-elastic behaviour
    eps_a * E = sa, which gives the stress-based form sqrt(sigma_max * sa) used for
    stress-life data (the "SWT equivalent fully reversed stress amplitude", for example in
    Dowling, Mechanical Behavior of Materials). So log_swt is the conventional stress-based
    form, computed from NOMINAL stresses: it ignores local plasticity and, for notched
    specimens, the notch-root stress. It equals sa at R = -1, exceeds sa for tensile mean
    stress, and is smaller than sa for R < -1, where the SWT relation is least reliable.

    log_swt is a fixed function of sa and R, so it adds no new information for a tree
    model; it is there to help the linear model represent the mean-stress effect. It is
    an input transformation, not a validated mean-stress correction for these alloys.
    """
    out = frame.copy()
    sa = out["stress_amplitude"].astype(float)
    r = out["R"].astype(float)
    out["log_sa"] = np.log10(sa)
    out["sigma_max"] = 2.0 * sa / (1.0 - r)
    out["sigma_mean"] = sa * (1.0 + r) / (1.0 - r)
    out["log_swt"] = np.log10(sa * np.sqrt(2.0 / (1.0 - r)))
    return out


# --------------------------------------------------------------------------- #
# Processing text -> heat-treatment class and surface class
# --------------------------------------------------------------------------- #
# The workbook stores post-processing as one text cell per curve, for example
#   "HIP 920 C, 2 hour, 1000 bar; SURF machine; SURF polish"
# Steps are separated by ';' and each step starts with one of four type tokens:
#   HT (heat treatment), HIP (hot isostatic pressing), NHT (no heat treatment),
#   SURF (a surface treatment or surface state).

def split_processing_steps(text) -> list[str]:
    """Split the processing cell into its steps. Empty or missing -> []."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or pd.isna(text):
        return []
    return [step.strip() for step in str(text).split(";") if step.strip()]


def heat_treatment_class(text) -> str:
    """Reduce the processing text to one of three heat-treatment classes.

    Rule (first match wins):
      1. any step of type HIP                      -> "HIP"
      2. any step of type HT                       -> "HT"
      3. otherwise (NHT stated, or no step given)  -> "NHT_or_unreported"

    The dataset authors filled unreported heat treatment with the default
    "NHT", so "NHT" cannot be told apart from "not reported"; curves with no
    heat-treatment step at all are put in the same honest class.
    """
    types = [step.split(" ")[0] for step in split_processing_steps(text)]
    if "HIP" in types:
        return "HIP"
    if "HT" in types:
        return "HT"
    return "NHT_or_unreported"


def heat_treatment_evidence(text) -> str:
    """Audit helper: what the cell literally says about heat treatment."""
    types = [step.split(" ")[0] for step in split_processing_steps(text)]
    if "HIP" in types and "HT" in types:
        return "HIP and HT steps"
    if "HIP" in types:
        return "HIP step"
    if "HT" in types:
        return "HT step"
    if "NHT" in types:
        return "NHT stated (may be a default)"
    return "no heat-treatment step in cell"


# Keyword rules for one SURF step, written as regular expressions and tested on the
# lower-cased step. Most keywords are word stems on purpose ("polish" must also find
# "electropolish", "blast" must find "sandblast"). Short words that could hide inside
# other words are anchored at a word boundary: "roll" must not match "controlled",
# "turn" must not match "return". The class priority is applied afterwards in `surface_class`.
SURFACE_KEYWORDS = {
    "peened": (r"peen", r"ultrasonic nanocrystal", r"hammer", r"\broll", r"attrition", r"mechanical impact"),
    "polished": (r"polish", r"finish", r"linish", r"tumble", r"fluidized bed"),
    "machined": (r"machine", r"\bedm\b", r"grind", r"\bcut\b", r"\bmill", r"\bturn"),
    "blasted": (r"blast",),
    "as-built": (r"as-built",),
}

# When a curve lists several surface steps, the class is the highest-priority
# one present. The order follows how strongly each treatment changes the
# surface that a fatigue crack starts from: peening (compressive residual
# stress) > polishing > machining > blasting > other > as-built.
SURFACE_PRIORITY = ["peened", "polished", "machined", "blasted", "other", "as-built"]


def classify_surface_step(step: str) -> str:
    """Map one 'SURF ...' step to a raw class using the keyword table."""
    text = step.lower()
    for cls in ("peened", "polished", "machined", "blasted", "as-built"):
        if any(re.search(pattern, text) for pattern in SURFACE_KEYWORDS[cls]):
            return cls
    return "other"


def surface_class(text) -> str:
    """Reduce the processing text to one surface class.

    Classes: "peened", "polished", "machined", "blasted", "other",
    "as-built_or_unreported". A curve with no SURF step gets the last class,
    because the dataset default for an unreported surface is "as-built".
    """
    steps = [s for s in split_processing_steps(text) if s.split(" ")[0] == "SURF"]
    if not steps:
        return "as-built_or_unreported"
    found = {classify_surface_step(step) for step in steps}
    for cls in SURFACE_PRIORITY:
        if cls in found:
            return "as-built_or_unreported" if cls == "as-built" else cls
    return "other"


def surface_evidence(text) -> str:
    """Audit helper: whether the cell states a surface step at all."""
    steps = [s for s in split_processing_steps(text) if s.split(" ")[0] == "SURF"]
    if not steps:
        return "no SURF step in cell"
    if all("as-built" in s.lower() for s in steps):
        return "as-built stated (may be a default)"
    return "surface treatment stated"


# --------------------------------------------------------------------------- #
# Other categorical rules
# --------------------------------------------------------------------------- #

def test_type_class(text) -> str:
    """Reduce the ten test-type strings to three loading classes.

    'uniaxial', 'uniaxial; VHCF', 'uniaxial; resonance', 'uniaxial; compression' -> "axial"
    'rotating bending fatigue'                                                   -> "rotating_bending"
    'bending fatigue', '3-point bending', '4-point bending' (+ VHCF/resonance)   -> "bending"
    """
    if pd.isna(text):
        return "unknown"
    lowered = str(text).lower()
    if "rotating" in lowered:
        return "rotating_bending"
    if "bending" in lowered:
        return "bending"
    if "uniaxial" in lowered:
        return "axial"
    return "other"


def build_direction_class(low, high, kind) -> str:
    """Class of the build direction (angle between specimen axis and build plate).

    The workbook gives the angle in degrees: 0 = horizontal, 90 = vertical.
    Two-value cells are bounds and are labelled "mixed"; empty cells are "unknown".
    """
    if kind == "missing" or kind == "unparsed":
        return "unknown"
    if kind == "range":
        return "mixed"
    if low <= 15:
        return "horizontal_0"
    if low >= 75:
        return "vertical_90"
    return "inclined"


# Alloy family is used for descriptive tables and plots only (not a model input).
_FAMILY_RULES = [
    # (family, regular expressions tested on the lower-cased material name; first match wins)
    ("Other", (r"^niti$", r"^we43$", r"^co", r"^cu", r"functionally graded")),
    ("Ti alloy", (r"^ti", r"^ta\d", r"^tc\d")),
    ("Al alloy", (r"^al", r"^a35\d", r"aluminum", r"scalmalloy", r"questek al", r"^ad1$", r"^aif357$")),
    ("Ni alloy", (r"^in\d", r"hastelloy", r"^gh\d", r"^k536$", r"^mar-m", r"^cmsx", r"^incol")),
    ("Steel", (r"steel", r"316", r"304", r"17-4", r"15-5", r"18ni", r"maraging", r"^ss\d", r"crni", r"^cp fe$", r"^sg2")),
]


def alloy_family(name) -> str:
    """Coarse alloy family from the material name, by keyword. Descriptive use only."""
    text = str(name).strip().lower()
    for family, patterns in _FAMILY_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return family
    return "Other"


# --------------------------------------------------------------------------- #
# Modelling matrix helpers
# --------------------------------------------------------------------------- #

def feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the model input columns, after checking none is forbidden."""
    assert_features_allowed(features)
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise KeyError(f"Feature columns missing from the dataset: {missing}")
    return frame[features].copy()

# --------------------------------------------------------------------------- #
# Test-regime columns (supplementary investigation B, version 3)
# --------------------------------------------------------------------------- #
# Fixed rules on the original test-method strings saved by notebook 02 (test_type_raw, load_control,
# frequency_text). They use only the description of how a test was run, never its outcome, so they can be applied
# before the train/test split. They are not part of any primary feature tier.

ULTRASONIC_MIN_HZ = 15000.0      # every recorded value of 1 kHz or above in the cohort lies between 19 and 20.5 kHz
KILOHERTZ_MIN_HZ = 1000.0


def frequency_bounds(text) -> tuple[float, float]:
    """Lowest and highest number in a frequency cell ("20000", "19500, 20500"); (nan, nan) if none."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan, np.nan
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(text))]
    return (min(numbers), max(numbers)) if numbers else (np.nan, np.nan)


def add_test_regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the test-regime columns used by supplementary investigation B.

    vhcf_tag      "VHCF" if the test-type string carries the database's VHCF tag, else "none"
    regime_tag    "VHCF", "resonance" or "none" from the test-type string (the resonance tag is incomplete:
                  many tests on resonance machines are untagged)
    khz_test      a recorded frequency (any bound) of 1 kHz or above
    ultrasonic    a recorded frequency (any bound) of 15 kHz or above
    displacement_controlled  load control recorded as "displacement" (the database default for VHCF tests)
    """
    out = frame.copy()
    test_type = out["test_type_raw"].astype(str)
    out["vhcf_tag"] = np.where(test_type.str.contains("VHCF", regex=False), "VHCF", "none")
    out["regime_tag"] = np.where(test_type.str.contains("VHCF", regex=False), "VHCF",
                                 np.where(test_type.str.contains("resonance", regex=False), "resonance", "none"))
    bounds = out["frequency_text"].map(frequency_bounds)
    high = bounds.map(lambda b: b[1])
    out["khz_test"] = (high >= KILOHERTZ_MIN_HZ).to_numpy()
    out["ultrasonic"] = (high >= ULTRASONIC_MIN_HZ).to_numpy()
    out["displacement_controlled"] = (out["load_control"].astype(str) == "displacement").to_numpy()
    return out
