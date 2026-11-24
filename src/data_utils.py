"""Data loading, hashing, audit tables and join checks for FatigueData-AM2022.

Every function here reads the raw workbook as it is. Nothing in this module
modifies a measured value or drops a row; it only loads, describes and joins.

Column names are discovered from the file, not assumed. The workbook headers
contain line breaks and Greek letters (for example "stress amplitude\\r\\nσa (MPa)"),
so `clean_column_name` turns them into short snake_case names and
`COLUMN_RENAMES` records the mapping for the audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
RAW_XLSX = RAW_DIR / "FatigueData-AM2022.xlsx"
RAW_JSON = RAW_DIR / "FatigueData-AM2022.json"

SHEET_SN = "S-N"
SHEET_EN = "e-N"
SHEET_DADN = "dadn"
SHEET_PARAM = "parameter"


# --------------------------------------------------------------------------- #
# File integrity
# --------------------------------------------------------------------------- #

def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file (Figshare publishes MD5 per file)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_raw_file_against_provenance(provenance_path: Path = AUDIT_DIR / "file_provenance.json") -> dict:
    """Recompute the hashes of the raw files and compare with the provenance record."""
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf8"))
    report = {}
    for name, info in provenance["files_downloaded"].items():
        path = RAW_DIR / name
        if not path.exists():
            report[name] = {"exists": False}
            continue
        sha = sha256_of_file(path)
        report[name] = {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256_matches_provenance": sha == info["sha256_on_disk"],
        }
    return report


# --------------------------------------------------------------------------- #
# Column name cleaning
# --------------------------------------------------------------------------- #

_GREEK = {"σ": "sigma", "ε": "epsilon", "Δ": "delta", "μ": "u", "°": "deg"}


def clean_column_name(name: str) -> str:
    """Turn a workbook header into a short snake_case name.

    Examples
    --------
    'stress amplitude\\r\\nσa (MPa)' -> 'stress_amplitude_sigma_a_mpa'
    'direciton of specimen \\r\\n(°)'  -> 'direciton_of_specimen_deg'  (typo kept, it is the file's)
    """
    text = str(name)
    for greek, latin in _GREEK.items():
        text = text.replace(greek, latin + "_")
    text = text.replace("'", "")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


# Short, readable aliases for the columns the project uses most. The left-hand
# side is the cleaned workbook header; the right-hand side is the project name.
# The mapping is applied *after* `clean_column_name`, so any header that is not
# listed here keeps its cleaned name.
COLUMN_RENAMES = {
    # S-N sheet
    "dataset_id": "dataset_id",
    "life_n_cycle": "N",
    "stress_amplitude_sigma_a_mpa": "stress_amplitude",
    "runout": "runout",
    # e-N sheet
    "strain_amplitude_epsilon_a_mm_mm": "strain_amplitude",
    # dadn sheet
    "stress_intensity_factor_range_delta_k_mpa_m_1_2": "delta_k",
    "crack_growth_rate_da_dn_m_cycle": "dadn",
    # parameter sheet
    "source_of_the_publication": "journal",
    "year_of_publication": "year",
    "country_and_region": "country",
    "funding_agency": "funding",
    "doi": "doi",
    "types_of_fatigue_data": "fatigue_data_type",
    "method_of_extraction": "extraction_method",
    "name_of_the_material": "material",
    "types_of_am": "am_process",
    "am_machine": "am_machine",
    "power_w": "power_w",
    "scan_speed_mm_s": "scan_speed_mm_s",
    "hatch_space_u_m": "hatch_space_um",
    "layer_thickness_u_m": "layer_thickness_um",
    "preheat_temperature_deg_c": "preheat_temperature_c",
    "direciton_of_specimen_deg": "build_direction_deg",
    "am_environment": "am_environment",
    "scan_pattern": "scan_pattern",
    "layer_scan_rotation_deg": "layer_rotation_deg",
    "voltage_v": "voltage_v",
    "current_a": "current_a",
    "speed_function": "speed_function",
    "powder_feed_rate_g_s": "powder_feed_rate_g_s",
    "wire_feed_rate_mm_s": "wire_feed_rate_mm_s",
    "types_of_feedstock": "feedstock_type",
    "size_of_feedstock_u_m": "feedstock_size_um",
    "processing_sequence_and_parameters": "processing",
    "types_of_fatigue_tests": "test_type",
    "fatigue_temperature_deg_c": "test_temperature_c",
    "fatigue_environment": "test_environment",
    "load_ratio": "R",
    "frequency_hz": "frequency_hz",
    "fatigue_machine": "test_machine",
    "fatigue_standard": "test_standard",
    "load_control": "load_control",
    "specimens_description": "specimen_description",
    "critical_cross_section_size_of_specimens_mm": "specimen_size_mm",
    "stress_concentration_factor_of_specimens": "kt",
    "youngs_modulus_gpa": "modulus_gpa",
    "yield_strength_mpa": "yield_strength_mpa",
    "ultimate_tensile_strength_mpa": "uts_mpa",
    "elongation": "elongation_pct",
    "rating_score": "rating_score",
}


def rename_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean every header and apply the project aliases. Returns (frame, mapping)."""
    mapping = {}
    new_names = []
    for original in frame.columns:
        cleaned = clean_column_name(original)
        final = COLUMN_RENAMES.get(cleaned, cleaned)
        mapping[str(original)] = final
        new_names.append(final)
    out = frame.copy()
    out.columns = new_names
    return out, mapping


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_sheet_raw(sheet: str, path: Path = RAW_XLSX) -> pd.DataFrame:
    """Load one sheet with the workbook's own headers, as text where it is text.

    The `parameter` sheet has two header rows: a group row ("metadata",
    "fatigue", ...) and the real column row. Row index 1 is used as the header
    for that sheet; the data sheets have a single header row.
    """
    header_row = 1 if sheet == SHEET_PARAM else 0
    return pd.read_excel(path, sheet_name=sheet, header=header_row, engine="openpyxl")


def load_sheet(sheet: str, path: Path = RAW_XLSX) -> tuple[pd.DataFrame, dict]:
    """Load one sheet and rename its columns to project names."""
    raw = load_sheet_raw(sheet, path)
    return rename_columns(raw)


def load_all_sheets(path: Path = RAW_XLSX) -> dict[str, pd.DataFrame]:
    """Load the four sheets (renamed) into a dict keyed by sheet name."""
    return {sheet: load_sheet(sheet, path)[0] for sheet in (SHEET_SN, SHEET_EN, SHEET_DADN, SHEET_PARAM)}


# --------------------------------------------------------------------------- #
# Audit tables
# --------------------------------------------------------------------------- #

def sheet_audit_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column audit: dtype, non-null count, missing share, unique count, example."""
    rows = []
    for col in frame.columns:
        series = frame[col]
        example = series.dropna().iloc[0] if series.notna().any() else None
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing_pct": round(100 * series.isna().mean(), 2),
                "n_unique": int(series.nunique(dropna=True)),
                "example": str(example)[:60] if example is not None else "",
            }
        )
    return pd.DataFrame(rows)


def sheet_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per sheet: rows, columns, fully duplicated rows."""
    rows = []
    for name, frame in frames.items():
        rows.append(
            {
                "sheet": name,
                "rows": len(frame),
                "columns": frame.shape[1],
                "fully_duplicated_rows": int(frame.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def sn_sheet_checks(sn: pd.DataFrame) -> dict:
    """Range and plausibility checks on the S-N sheet (section 9 of 03_data_understanding)."""
    sa = pd.to_numeric(sn["stress_amplitude"], errors="coerce")
    n = pd.to_numeric(sn["N"], errors="coerce")
    runout = pd.to_numeric(sn["runout"], errors="coerce")
    return {
        "rows": int(len(sn)),
        "unique_dataset_ids": int(sn["dataset_id"].nunique()),
        "runout_value_counts": {str(k): int(v) for k, v in runout.value_counts(dropna=False).items()},
        "runout_count": int((runout == 1).sum()),
        "runout_share_pct": round(100 * float((runout == 1).mean()), 2),
        "stress_amplitude_min": float(sa.min()),
        "stress_amplitude_max": float(sa.max()),
        "N_min": float(n.min()),
        "N_max": float(n.max()),
        "count_sa_le_0": int((sa <= 0).sum()),
        "count_N_le_0": int((n <= 0).sum()),
        "count_N_lt_10": int((n < 10).sum()),
        "count_N_gt_1e10": int((n > 1e10).sum()),
        "count_sa_missing": int(sa.isna().sum()),
        "count_N_missing": int(n.isna().sum()),
        "count_non_finite": int((~np.isfinite(sa.fillna(np.nan)) | ~np.isfinite(n.fillna(np.nan))).sum()),
        "duplicated_id_sa_N_rows": int(sn.duplicated(subset=["dataset_id", "stress_amplitude", "N"]).sum()),
    }


# --------------------------------------------------------------------------- #
# Join checks
# --------------------------------------------------------------------------- #

def join_check(sn: pd.DataFrame, param: pd.DataFrame, key: str = "dataset_id") -> dict:
    """Describe how the S-N sheet and the parameter sheet relate on `key`."""
    sn_ids = set(sn[key].dropna().unique())
    param_ids = set(param[key].dropna().unique())
    param_sn = param[param["fatigue_data_type"] == "sn"] if "fatigue_data_type" in param else param
    return {
        "key_unique_in_parameter": bool(param[key].is_unique),
        "parameter_rows": int(len(param)),
        "parameter_rows_type_sn": int(len(param_sn)),
        "sn_unique_ids": len(sn_ids),
        "sn_ids_not_in_parameter": sorted(int(i) for i in sn_ids - param_ids),
        "parameter_sn_ids_without_sn_rows": sorted(int(i) for i in set(param_sn[key]) - sn_ids),
        "parameter_ids_not_in_any_data_sheet": None,  # filled by the notebook when all sheets are loaded
    }


def join_sn_to_parameter(sn: pd.DataFrame, param: pd.DataFrame, key: str = "dataset_id") -> pd.DataFrame:
    """Left-join S-N rows to their curve metadata. Asserts the key is unique in `param`."""
    if not param[key].is_unique:
        raise ValueError(f"'{key}' is not unique in the parameter sheet; cannot join safely.")
    joined = sn.merge(param, on=key, how="left", validate="many_to_one", indicator=True)
    joined["_matched"] = joined["_merge"] == "both"
    return joined.drop(columns="_merge")


# --------------------------------------------------------------------------- #
# Article identifier
# --------------------------------------------------------------------------- #

def normalise_title(title) -> str:
    """Lower-case, strip punctuation and collapse spaces, for use as a fallback article key."""
    if pd.isna(title):
        return ""
    text = re.sub(r"[^0-9a-z]+", " ", str(title).lower())
    return re.sub(r"\s+", " ", text).strip()


def build_article_id(param: pd.DataFrame) -> pd.Series:
    """Article identifier: the DOI (lower-cased, stripped) when present, else the normalised title.

    The identifier is used for grouping and auditing only; it is never a feature.
    """
    doi = param["doi"].astype("string").str.strip().str.lower()
    doi = doi.where(doi.notna() & (doi != "") & (doi != "nan"))
    title_key = param["title"].map(normalise_title)
    from_title = ("title:" + title_key).where(title_key != "")   # stays missing when there is no title either
    return doi.fillna(from_title).astype("string")


# --------------------------------------------------------------------------- #
# Two-value numeric cells
# --------------------------------------------------------------------------- #

def parse_numeric_cell(value) -> tuple[float | None, float | None, str]:
    """Parse a numeric workbook cell that may hold one value or two bounds.

    The dataset descriptor says: one value is a value or a mean; two values are a
    lower and an upper bound. Returns (low, high, kind) with kind in
    {"missing", "single", "range", "unparsed"}. A single value is returned as
    (v, v, "single"). Callers decide what to do with ranges; nothing is averaged here.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None, "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), float(value), "single"
    text = str(value).strip().replace("\u2212", "-")   # a typographic minus sign is read as a minus
    if text == "" or text.lower() == "nan":
        return None, None, "missing"
    parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip() != ""]
    numbers = []
    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError:
            return None, None, "unparsed"
    if len(numbers) == 1:
        return numbers[0], numbers[0], "single"
    if len(numbers) == 2:
        return min(numbers), max(numbers), "range"
    return None, None, "unparsed"


def parse_numeric_column(series: pd.Series) -> pd.DataFrame:
    """Apply `parse_numeric_cell` to a column. Returns columns low, high, kind."""
    parsed = series.map(parse_numeric_cell)
    out = pd.DataFrame(parsed.tolist(), index=series.index, columns=["low", "high", "kind"])
    return out


def single_value_or_nan(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (value, kind): the numeric value when the cell holds exactly one number, else NaN.

    Two-value cells (bounds) become NaN so that nothing is averaged silently; the
    `kind` series lets the notebook count how many cells were ranges.
    """
    parsed = parse_numeric_column(series)
    value = parsed["low"].where(parsed["kind"] == "single", np.nan).astype(float)
    return value, parsed["kind"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def save_json(obj, path: Path) -> None:
    """Write JSON with UTF-8 and indentation; numpy types are converted."""

    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        raise TypeError(f"Not serialisable: {type(value)}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=convert, ensure_ascii=False), encoding="utf8")


def count_points_curves_articles(frame: pd.DataFrame, curve_col: str = "dataset_id", article_col: str = "article_id") -> dict:
    """Return the number of points (rows), curves and articles in a frame."""
    return {
        "points": int(len(frame)),
        "curves": int(frame[curve_col].nunique()),
        "articles": int(frame[article_col].nunique()) if article_col in frame else None,
    }


# --------------------------------------------------------------------------- #
# Data acquisition (official Figshare record only)
# --------------------------------------------------------------------------- #

FIGSHARE_ARTICLE_ID = 22337629
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"


def ensure_raw_data(file_name: str = "FatigueData-AM2022.xlsx") -> Path:
    """Return the path of the raw file, downloading it from Figshare if it is absent.

    The file list is read from the public Figshare API record of the dataset
    (DOI 10.6084/m9.figshare.22337629) and the file is fetched from the URL the
    API returns. No other source is used. The MD5 that Figshare publishes is
    checked after the download. An existing file is never overwritten.
    """
    import urllib.request

    target = RAW_DIR / file_name
    if target.exists():
        return target
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(FIGSHARE_API_URL, timeout=60) as response:
        record = json.loads(response.read().decode("utf8"))
    matches = [f for f in record["files"] if f["name"] == file_name]
    if not matches:
        raise FileNotFoundError(f"{file_name} is not listed in the Figshare record {FIGSHARE_API_URL}")
    info = matches[0]
    urllib.request.urlretrieve(info["download_url"], target)
    if md5_of_file(target) != info["computed_md5"]:
        target.unlink()
        raise IOError(f"MD5 mismatch after downloading {file_name}; the partial file was removed.")
    return target


DESCRIPTOR_PAPER = ("Zhang, Z. & Xu, Z. Fatigue database of additively manufactured alloys. "
                    "Scientific Data 10:249 (2023). https://doi.org/10.1038/s41597-023-02150-x")
PROVENANCE_PATH = AUDIT_DIR / "file_provenance.json"
API_RECORD_PATH = AUDIT_DIR / "figshare_api_record.json"


def fetch_figshare_record() -> dict:
    """The public Figshare API record of the dataset (network access needed)."""
    import urllib.request
    with urllib.request.urlopen(FIGSHARE_API_URL, timeout=60) as response:
        return json.loads(response.read().decode("utf8"))


def build_provenance_record(api_record: dict, file_names: list[str], raw_dir: Path = RAW_DIR,
                            download_date_utc: str | None = None) -> dict:
    """Provenance record of the raw files, built from the Figshare API record and the files on disk.

    Pure function (no network): every field comes from `api_record` or from hashing the files. When the date of
    the download is not known, the modification time of the oldest raw file is used and labelled as such.
    """
    from datetime import datetime, timezone
    listed = {f["name"]: f for f in api_record["files"]}
    files = {}
    for name in file_names:
        path = Path(raw_dir) / name
        info = listed[name]
        md5 = md5_of_file(path)
        files[name] = {
            "figshare_file_id": info["id"],
            "download_url": info["download_url"],
            "size_bytes_reported_by_figshare": info["size"],
            "size_bytes_on_disk": path.stat().st_size,
            "md5_reported_by_figshare": info["computed_md5"],
            "md5_on_disk": md5,
            "md5_match": md5 == info["computed_md5"],
            "sha256_on_disk": sha256_of_file(path),
        }
    if download_date_utc is None:
        oldest = min((Path(raw_dir) / n).stat().st_mtime for n in file_names)
        download_date_utc = datetime.fromtimestamp(oldest, tz=timezone.utc).isoformat(timespec="seconds")
        date_source = "modification time of the raw files on disk"
    else:
        date_source = "recorded at download"
    doi = api_record.get("doi", "")
    return {
        "dataset_title": api_record.get("title"),
        "dataset_doi": doi,
        "dataset_doi_versionless": doi.rsplit(".v", 1)[0] if ".v" in doi else doi,
        "landing_page": api_record.get("figshare_url"),
        "api_record_url": FIGSHARE_API_URL,
        "figshare_version": api_record.get("version"),
        "figshare_published_date": api_record.get("published_date"),
        "license": api_record.get("license"),
        "creators": [a.get("full_name") for a in api_record.get("authors", [])],
        "descriptor_paper": DESCRIPTOR_PAPER,
        "download_date_utc": download_date_utc,
        "download_date_source": date_source,
        "download_method": "Figshare public API file list, then download of the API download_url (no other source used)",
        "files_downloaded": files,
        "files_available_but_not_downloaded": sorted(set(listed) - set(file_names)),
        "raw_data_policy": "Files in data/raw are never modified. All derived data are written to data/processed.",
        "record_created_by": "src/data_utils.ensure_provenance_record",
    }


def ensure_provenance_record(file_names: list[str], provenance_path: Path = PROVENANCE_PATH,
                             api_record_path: Path = API_RECORD_PATH) -> dict:
    """Return the provenance record of the raw files, creating it if it does not exist.

    An existing record is never overwritten: it documents the original download, and every later run is checked
    against it. In a checkout without the record (for example after `results/` was cleaned), the Figshare API record
    is fetched (or read from `api_record_path` if present), the record is built from it and from the files on disk,
    and both are saved.
    """
    provenance_path, api_record_path = Path(provenance_path), Path(api_record_path)
    if provenance_path.exists():
        return json.loads(provenance_path.read_text(encoding="utf8"))
    if api_record_path.exists():
        api_record = json.loads(api_record_path.read_text(encoding="utf8"))
    else:
        api_record = fetch_figshare_record()
        api_record_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(api_record, api_record_path)
    record = build_provenance_record(api_record, file_names)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(record, provenance_path)
    return record


# --------------------------------------------------------------------------- #
# Near-duplicate curves and linked groups of articles
# --------------------------------------------------------------------------- #

def find_near_duplicate_curves(frame: pd.DataFrame, curve_col: str = "dataset_id", article_col: str = "article_id",
                               alloy_col: str = "material", stress_col: str = "stress_amplitude", life_col: str = "N",
                               stress_tol: float = 0.01, life_tol: float = 0.02,
                               min_shared: int = 4, min_share: float = 0.8,
                               same_alloy: bool = True, across_articles: bool = True) -> pd.DataFrame:
    """Find pairs of curves whose points almost coincide.

    A point of one curve "matches" another curve when that curve has a point within
    `stress_tol` in log10(stress) (0.01 is about 2%) and `life_tol` in log10(life)
    (0.02 is about 5%). Two curves are flagged when at least `min_shared` points of the
    smaller curve match, and these are at least `min_share` of the smaller curve. When the two curves have the
    same number of points, the matches are counted in both directions and the larger count is used.

    With `across_articles=True` (default) only curves from different articles are
    compared: such pairs are the same test data printed in two publications and
    digitised twice, not independent evidence. With `across_articles=False` only
    curves inside one article are compared (the same data stored under two ids).
    With `same_alloy=True` the two curves must carry the same alloy name.

    Method: points are put on a grid whose cells are one tolerance wide, each point is
    compared with the points in its own and the eight neighbouring cells, and the true
    distance is then checked. (Comparing grid cells alone would miss two close points
    that fall on either side of a cell edge.)
    """
    points = pd.DataFrame({
        "curve": frame[curve_col].to_numpy(),
        "article": frame[article_col].to_numpy(),
        "alloy": frame[alloy_col].astype(str).to_numpy(),
        "ls": np.log10(frame[stress_col].astype(float).to_numpy()),
        "ln": np.log10(frame[life_col].astype(float).to_numpy()),
    })
    points["point"] = np.arange(len(points))
    points["cell_s"] = np.floor(points["ls"] / stress_tol).astype(int)
    points["cell_n"] = np.floor(points["ln"] / life_tol).astype(int)
    other = points.add_suffix("_b")

    matches = []
    for shift_s in (-1, 0, 1):
        for shift_n in (-1, 0, 1):
            left = points.assign(cell_s_b=points["cell_s"] + shift_s, cell_n_b=points["cell_n"] + shift_n)
            pairs = left.merge(other, on=["cell_s_b", "cell_n_b"])
            keep = (pairs["curve"] != pairs["curve_b"])
            keep &= (pairs["ls"] - pairs["ls_b"]).abs() <= stress_tol
            keep &= (pairs["ln"] - pairs["ln_b"]).abs() <= life_tol
            keep &= (pairs["article"] != pairs["article_b"]) if across_articles else (pairs["article"] == pairs["article_b"])
            if same_alloy:
                keep &= pairs["alloy"] == pairs["alloy_b"]
            matches.append(pairs.loc[keep, ["point", "curve", "article", "alloy", "curve_b", "article_b", "alloy_b"]])
    matches = pd.concat(matches, ignore_index=True).drop_duplicates(["point", "curve_b"])

    # For every ordered pair (curve, curve_b): how many points of `curve` have a partner in `curve_b`?
    counts = (matches.groupby(["curve", "curve_b"])
              .agg(shared_points=("point", "nunique"), article=("article", "first"), article_b=("article_b", "first"),
                   alloy=("alloy", "first"), alloy_b=("alloy_b", "first")).reset_index())
    sizes = points.groupby("curve").size()
    counts["size"] = sizes.loc[counts["curve"]].to_numpy()
    counts["size_b"] = sizes.loc[counts["curve_b"]].to_numpy()
    # Judge each pair from the side of its smaller curve. When both curves have the same number of points, count
    # the matches in both directions and keep the larger count, so that the result does not depend on curve order.
    tie = counts["size"] == counts["size_b"]
    unordered = np.where(counts["curve"] < counts["curve_b"], counts["curve"].astype(str) + "|" + counts["curve_b"].astype(str),
                         counts["curve_b"].astype(str) + "|" + counts["curve"].astype(str))
    counts["pair_key"] = unordered
    best_tie = counts[tie].sort_values(["pair_key", "shared_points", "curve"], ascending=[True, False, True]).drop_duplicates("pair_key")
    counts = pd.concat([counts[~tie & (counts["size"] < counts["size_b"])], best_tie]).drop(columns="pair_key")
    counts["shared_share"] = counts["shared_points"] / counts["size"]
    flagged = counts[(counts["shared_points"] >= min_shared) & (counts["shared_share"] >= min_share)].copy()

    first_is_a = (flagged["curve"] < flagged["curve_b"]).to_numpy()
    result = pd.DataFrame({
        f"{curve_col}_a": np.where(first_is_a, flagged["curve"], flagged["curve_b"]),
        f"{curve_col}_b": np.where(first_is_a, flagged["curve_b"], flagged["curve"]),
        f"{article_col}_a": np.where(first_is_a, flagged["article"], flagged["article_b"]),
        f"{article_col}_b": np.where(first_is_a, flagged["article_b"], flagged["article"]),
        "alloy": np.where(first_is_a, flagged["alloy"], flagged["alloy_b"]),
        "alloy_b": np.where(first_is_a, flagged["alloy_b"], flagged["alloy"]),
        "shared_points": flagged["shared_points"].to_numpy(),
        "smaller_curve_points": flagged["size"].to_numpy(),
        "shared_share": flagged["shared_share"].to_numpy(),
    })
    return result.sort_values([f"{curve_col}_a", f"{curve_col}_b"]).reset_index(drop=True)


def connected_groups(items, pairs) -> dict:
    """Group items that are linked directly or through a chain of pairs (union-find).

    Returns {item: group_label}; the label is the smallest item of the group, so
    the result does not depend on the order of the input.
    """
    parent = {item: item for item in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            low, high = sorted([root_a, root_b])
            parent[high] = low
    return {item: find(item) for item in parent}


def author_cluster_ids(article_authors: pd.Series) -> pd.Series:
    """Cluster articles that share at least one author name.

    `article_authors` is indexed by article id and holds the workbook's author
    string ("Last, First; Last, First; ..."). Two articles are linked when the
    same lower-cased name string appears in both; links are chained. This is an
    approximation of "same research group": different spellings of one person
    are missed, and two people with the same name are merged.
    """
    first_article_of_author = {}
    links = []
    for article, authors in article_authors.items():
        for name in str(authors).split(";"):
            key = name.strip().lower()
            if not key or key == "nan":
                continue
            if key in first_article_of_author:
                links.append((first_article_of_author[key], article))
            else:
                first_article_of_author[key] = article
    groups = connected_groups(list(article_authors.index), links)
    return pd.Series(groups, name="author_cluster_id").reindex(article_authors.index)


# --------------------------------------------------------------------------- #
# Filter funnel
# --------------------------------------------------------------------------- #

def temperature_evidence(frame: pd.DataFrame, temperature_col: str = "test_temperature_c",
                         article_col: str = "article_id", default_value: float = 25.0) -> pd.Series:
    """Classify what a recorded test temperature can be taken to mean.

    The dataset authors state that tests are assumed to be "conducted under an ambient
    environment (25 °C, air)" when nothing else is reported, and the temperature column has
    no empty cells. A recorded 25 °C can therefore mean a reported 25 °C, a reported "room
    temperature" without a number, or no information at all. The file cannot separate these.

    Returns one label per row:
      "explicit, not the default"        : a value other than 25 °C was recorded;
      "25, article reports other temps"  : 25 °C in an article whose S-N curves also record a
                                           temperature other than 25 °C (the article evidently
                                           reports temperature, so 25 °C is probably reported);
      "25, default or reported"          : 25 °C with no such evidence (ambiguous).
    `frame` must contain every S-N curve of each article (before temperature filtering), so
    that the article-level evidence is complete.
    """
    temp = frame[temperature_col].astype(float)
    reports_other = frame.loc[temp != default_value, article_col].unique()
    labels = np.where(temp != default_value, "explicit, not the default",
                      np.where(frame[article_col].isin(reports_other), "25, article reports other temps",
                               "25, default or reported"))
    return pd.Series(labels, index=frame.index, name="temperature_evidence")


def funnel_row(step: int, name: str, frame: pd.DataFrame, previous: pd.DataFrame | None = None,
               curve_col: str = "dataset_id", article_col: str = "article_id") -> dict:
    """One line of the filter funnel: what is left after a step and what the step removed."""
    row = {"step": step, "filter": name, "points": int(len(frame)),
           "curves": int(frame[curve_col].nunique()), "articles": int(frame[article_col].nunique())}
    if previous is not None:
        row["points_removed"] = int(len(previous) - len(frame))
        row["curves_removed"] = int(previous[curve_col].nunique() - frame[curve_col].nunique())
        row["articles_removed"] = int(previous[article_col].nunique() - frame[article_col].nunique())
    else:
        row.update({"points_removed": 0, "curves_removed": 0, "articles_removed": 0})
    return row
