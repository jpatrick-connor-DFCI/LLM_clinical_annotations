"""Structured PROFILE_DATA sources: PSA and testosterone labs, medications, vital status.

These feed the disease-course builders (tasks/bcr, tasks/progression,
tasks/mcrpc) alongside the note findings. Column names follow the parent
repo's ONCDRS contract (`oncdrs_dedup.DATA_COLUMNS`):

  LABS.parquet        DFCI_MRN, SPECIMEN_COLLECT_DT, TEST_TYPE_CD,
                      TEST_TYPE_DESCR, NUMERIC_RESULT, TEXT_RESULT, RESULT_UOM_NM
  MEDICATIONS.parquet DFCI_MRN, NCI_PREFERRED_MED_NM, MED_START_DT
  PT_INFO_STATUS_REGISTRATION.parquet
                      DFCI_MRN, HYBRID_DEATH_DT, DERIVED_LAST_ALIVE_DATE, ...

Only those columns are guaranteed. Optional identity and exposure columns
(specimen/accession ids, dose, route, days supply, stop dates) are kept when a
file has them and are otherwise unknown, never defaulted.

Every loader scans lazily with the MRN filter pushed into the scan, like
`preprocessing.notes.load_profile_notes`. No provider SDK is imported here.
"""

import json
import math
import re
from pathlib import Path

import polars as pl

from preprocessing.config import PROFILE_DATA_PATH
from preprocessing.notes import to_iso_date

STRUCTURED_FILES = {
    "labs": "LABS.parquet",
    "medications": "MEDICATIONS.parquet",
    "registry": "PT_INFO_STATUS_REGISTRATION.parquet",
}

# The parent repo's BROAD_PSA_CODES without CPSA, which is complexed PSA and
# not interchangeable with total PSA.
PSA_CODES = ("PSA", "PSAR", "PSATOTSCRN", "PSAMON", "PSAULT", "PSAT")
TESTOSTERONE_CODES = ("TES", "TESTT", "TOTTES", "TTEST")

# ONCDRS stores below-detection results as this NUMERIC_RESULT with "<..." in
# TEXT_RESULT (parent repo, dfci_labs.BELOW_DETECTION_SENTINEL).
BELOW_DETECTION_SENTINEL = 9999999.0
PSA_PLAUSIBLE = (0.0, 10_000.0)
TESTOSTERONE_PLAUSIBLE_NG_DL = (0.0, 2000.0)
NMOL_L_TO_NG_DL = 28.84

# Optional columns kept when present. The first present column in each group
# becomes the canonical field.
SPECIMEN_ID_COLUMNS = ("SPECIMEN_ID", "ACCESSION_NUMBER", "ACCESSION_NBR", "ORDER_ID", "RESULT_ID")
MEDICATION_OPTIONAL_COLUMNS = {
    "stop_date": ("MED_END_DT", "MED_STOP_DT", "MED_DISCONTINUE_DT"),
    "administration_date": ("MED_ADMIN_DT", "ADMIN_DT", "ADMINISTRATION_DT"),
    "dose": ("MED_DOSE", "DOSE", "DOSE_AMT"),
    "dose_unit": ("MED_DOSE_UNIT", "DOSE_UNIT", "DOSE_UOM"),
    "route": ("MED_ROUTE_NM", "ROUTE", "MED_ROUTE"),
    "formulation": ("MED_FORM", "FORMULATION", "MED_NM", "ORDER_MED_NM"),
    "days_supply": ("DAYS_SUPPLY", "MED_DAYS_SUPPLY"),
    "frequency": ("MED_FREQ", "FREQUENCY", "MED_FREQUENCY"),
}

# Medication classes. Names are matched as words in the upper-cased NCI
# preferred name, so salt forms ("LEUPROLIDE ACETATE") map to their agent.
# Lists follow the parent repo's compile_COMPASS_cohort_data.py and
# classify_adt_intent.py, except that relugolix is oral ADT here: the parent's
# DEPOT_ADT_MEDS includes it only because it is not an antiandrogen.
MEDICATION_AGENTS = {
    "depot_adt": ("LEUPROLIDE", "GOSERELIN", "TRIPTORELIN", "HISTRELIN", "DEGARELIX"),
    "oral_adt": ("RELUGOLIX",),
    "antiandrogen": ("BICALUTAMIDE", "FLUTAMIDE", "NILUTAMIDE"),
    "arpi": ("ABIRATERONE", "ENZALUTAMIDE", "APALUTAMIDE", "DAROLUTAMIDE"),
    "taxane": ("DOCETAXEL", "CABAZITAXEL"),
    "radium": ("RADIUM",),
    "lu_psma": ("LUTETIUM", "VIPIVOTIDE", "PLUVICTO"),
    "sipuleucel": ("SIPULEUCEL",),
    "parpi": ("OLAPARIB", "RUCAPARIB", "NIRAPARIB", "TALAZOPARIB"),
}
# Single-agent platinum only, matching the parent's exact-name endpoint.
PLATINUM_EXACT = {"CARBOPLATIN": "carboplatin", "CISPLATIN": "cisplatin"}
_AGENT_PATTERNS = [
    (drug_class, agent, re.compile(rf"\b{agent}\b"))
    for drug_class, agents in MEDICATION_AGENTS.items()
    for agent in agents
]
_LU_PSMA_AGENT = "lutetium_psma"

# Registry reduction policy. Bump on any change; builders record it.
VITAL_STATUS_POLICY = "registry-max-v1"

_BOUND = re.compile(r"^\s*(<=|>=|≤|≥|<|>)\s*(\d+(?:\.\d+)?|\.\d+)")
_PLAIN_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*(?:[a-z/%]+(?:\s*/\s*[a-z]+)?)?\s*$", re.I)
_UNDETECTABLE_TEXT = re.compile(
    r"\b(?:undetect\w*|not\s+detected|non[\s-]?detect\w*|below\s+(?:the\s+)?(?:limit|detection))\b",
    re.IGNORECASE,
)
_COMPARATOR = {"<": "<", ">": ">", "<=": "<=", ">=": ">=", "≤": "<=", "≥": ">="}


def source_path(name, root=None):
    return Path(root or PROFILE_DATA_PATH) / STRUCTURED_FILES[name]


def structured_source_fingerprint(paths):
    """Path, size and mtime of each structured source ({name: path}).

    The files are too large to hash on every builder run; a changed size or
    mtime is enough to show the inputs moved.
    """
    record = {}
    for name, path in sorted(paths.items()):
        path = Path(path)
        if path.exists():
            stat = path.stat()
            record[name] = {"path": str(path), "size": stat.st_size, "mtime": int(stat.st_mtime)}
        else:
            record[name] = {"path": str(path), "missing": True}
    return json.dumps(record, sort_keys=True)


def _scan(path, mrns):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structured source not found: {path}")
    lazy = pl.scan_parquet(path).with_row_index("source_row")
    if mrns is not None:
        lazy = lazy.filter(
            pl.col("DFCI_MRN").cast(pl.Int64, strict=False).is_in(sorted(int(m) for m in mrns))
        )
    return lazy


def _frame(rows, schema):
    return pl.DataFrame({c: [row.get(c) for row in rows] for c in schema}, schema=schema, strict=False)


def _first_present(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_lab_result(numeric, text):
    """Return (value, comparator) for one lab row, or None when unusable.

    comparator is "=", "<", "<=", ">", ">=", or "undetectable" (below
    detection with no recoverable limit; value is then None). A bound is never
    turned into an exact value.
    """
    text = "" if text is None else str(text).strip()
    numeric = _finite(numeric)
    bound = _BOUND.match(text)
    if bound:
        return float(bound.group(2)), _COMPARATOR[bound.group(1)]
    if numeric is not None and numeric >= BELOW_DETECTION_SENTINEL:
        if text.startswith("<") or _UNDETECTABLE_TEXT.search(text):
            return None, "undetectable"
        return None
    if numeric is not None:
        return numeric, "="
    plain = _PLAIN_NUMBER.match(text)
    if plain:
        return float(plain.group(1)), "="
    if _UNDETECTABLE_TEXT.search(text):
        return None, "undetectable"
    return None


def _lab_rows(path, mrns, codes):
    lazy = _scan(path, mrns)
    columns = set(lazy.collect_schema().names())
    required = {"DFCI_MRN", "SPECIMEN_COLLECT_DT", "TEST_TYPE_CD", "NUMERIC_RESULT", "TEXT_RESULT"}
    missing = required - columns
    if missing:
        raise ValueError(f"{path} is missing lab columns: {sorted(missing)}")
    specimen = _first_present(columns, SPECIMEN_ID_COLUMNS)
    frame = lazy.filter(
        pl.col("TEST_TYPE_CD").cast(pl.Utf8).str.strip_chars().str.to_uppercase().is_in(list(codes))
    ).select(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        pl.col("source_row").cast(pl.Int64),
        pl.col("SPECIMEN_COLLECT_DT").cast(pl.Utf8).alias("_date"),
        pl.col("TEST_TYPE_CD").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("test_code"),
        pl.col("NUMERIC_RESULT").cast(pl.Float64, strict=False).alias("_numeric"),
        pl.col("TEXT_RESULT").cast(pl.Utf8).alias("text_result"),
        (pl.col("RESULT_UOM_NM").cast(pl.Utf8) if "RESULT_UOM_NM" in columns
         else pl.lit(None, dtype=pl.Utf8)).alias("unit"),
        (pl.col(specimen).cast(pl.Utf8) if specimen else pl.lit(None, dtype=pl.Utf8)).alias("specimen_id"),
    ).collect()
    for row in frame.iter_rows(named=True):
        if row["DFCI_MRN"] is None:
            continue
        date = to_iso_date(row["_date"])
        parsed = parse_lab_result(row["_numeric"], row["text_result"])
        if date is None or parsed is None:
            continue
        yield row, date, parsed


PSA_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "date": pl.Utf8,
    "date_precision": pl.Utf8,
    "psa_ng_ml": pl.Float64,
    "comparator": pl.Utf8,
    "source": pl.Utf8,
    "source_id": pl.Utf8,
    "specimen_id": pl.Utf8,
    "test_code": pl.Utf8,
    "text_result": pl.Utf8,
}


def load_psa(mrns=None, path=None):
    """Total-PSA results from LABS, one row per source row (ng/mL)."""
    rows = []
    for row, date, (value, comparator) in _lab_rows(path or source_path("labs"), mrns, PSA_CODES):
        if value is not None and not PSA_PLAUSIBLE[0] <= value <= PSA_PLAUSIBLE[1]:
            continue
        rows.append({
            "DFCI_MRN": row["DFCI_MRN"],
            "date": date,
            "date_precision": "day",
            "psa_ng_ml": value,
            "comparator": comparator,
            "source": "labs",
            "source_id": f"labs:{row['source_row']}",
            "specimen_id": row["specimen_id"],
            "test_code": row["test_code"],
            "text_result": row["text_result"],
        })
    return _frame(rows, PSA_SCHEMA)


TESTOSTERONE_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "date": pl.Utf8,
    "testosterone_ng_dl": pl.Float64,
    "comparator": pl.Utf8,
    "unit": pl.Utf8,
    "unit_assumed": pl.Boolean,
    "source_id": pl.Utf8,
}


def _testosterone_factor(unit):
    """ng/dL multiplier for a unit string; None when unrecognized."""
    if unit is None or not str(unit).strip():
        return 1.0
    unit = re.sub(r"\s+", "", str(unit).lower())
    if "nmol" in unit:
        return NMOL_L_TO_NG_DL
    if unit in {"ng/dl", "ngdl"}:
        return 1.0
    if unit in {"ng/ml", "ngml"}:
        return 100.0
    return None


def load_testosterone(mrns=None, path=None):
    """Total testosterone from LABS in ng/dL. A missing unit is assumed ng/dL and flagged."""
    rows = []
    for row, date, (value, comparator) in _lab_rows(
        path or source_path("labs"), mrns, TESTOSTERONE_CODES
    ):
        factor = _testosterone_factor(row["unit"])
        if factor is None:
            continue
        if value is not None:
            value = value * factor
            if not TESTOSTERONE_PLAUSIBLE_NG_DL[0] <= value <= TESTOSTERONE_PLAUSIBLE_NG_DL[1]:
                continue
        rows.append({
            "DFCI_MRN": row["DFCI_MRN"],
            "date": date,
            "testosterone_ng_dl": value,
            "comparator": comparator,
            "unit": row["unit"],
            "unit_assumed": row["unit"] is None or not str(row["unit"]).strip(),
            "source_id": f"labs:{row['source_row']}",
        })
    return _frame(rows, TESTOSTERONE_SCHEMA)


def classify_medication(name):
    """(drug_class, agent) for an NCI preferred medication name, or None."""
    if name is None:
        return None
    upper = re.sub(r"\s+", " ", str(name).upper()).strip()
    if upper in PLATINUM_EXACT:
        return "platinum", PLATINUM_EXACT[upper]
    for drug_class, agent, pattern in _AGENT_PATTERNS:
        if pattern.search(upper):
            if drug_class == "lu_psma":
                return drug_class, _LU_PSMA_AGENT
            return drug_class, agent.lower()
    return None


MEDICATION_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "date": pl.Utf8,
    "drug_class": pl.Utf8,
    "agent": pl.Utf8,
    "med_name": pl.Utf8,
    "source_id": pl.Utf8,
    **{name: pl.Utf8 for name in MEDICATION_OPTIONAL_COLUMNS},
}


def load_medications(mrns=None, path=None):
    """Prostate-cancer-relevant medication records, one row per source row.

    Optional exposure fields are kept as text when present; `stop_date` and
    `administration_date` are normalized to ISO dates.
    """
    path = path or source_path("medications")
    lazy = _scan(path, mrns)
    columns = set(lazy.collect_schema().names())
    missing = {"DFCI_MRN", "NCI_PREFERRED_MED_NM", "MED_START_DT"} - columns
    if missing:
        raise ValueError(f"{path} is missing medication columns: {sorted(missing)}")
    optional = {
        name: _first_present(columns, candidates)
        for name, candidates in MEDICATION_OPTIONAL_COLUMNS.items()
    }
    frame = lazy.select(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        pl.col("source_row").cast(pl.Int64),
        pl.col("NCI_PREFERRED_MED_NM").cast(pl.Utf8).alias("med_name"),
        pl.col("MED_START_DT").cast(pl.Utf8).alias("_date"),
        *[
            (pl.col(column).cast(pl.Utf8) if column else pl.lit(None, dtype=pl.Utf8)).alias(name)
            for name, column in optional.items()
        ],
    ).collect()
    rows = []
    for row in frame.iter_rows(named=True):
        match = classify_medication(row["med_name"])
        date = to_iso_date(row["_date"])
        if row["DFCI_MRN"] is None or match is None or date is None:
            continue
        record = {
            "DFCI_MRN": row["DFCI_MRN"],
            "date": date,
            "drug_class": match[0],
            "agent": match[1],
            "med_name": row["med_name"],
            "source_id": f"medications:{row['source_row']}",
        }
        for name in MEDICATION_OPTIONAL_COLUMNS:
            value = row[name]
            if name in {"stop_date", "administration_date"}:
                value = to_iso_date(value)
            record[name] = value
        rows.append(record)
    return _frame(rows, MEDICATION_SCHEMA)


VITAL_STATUS_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "death_date": pl.Utf8,
    "death_date_conflict": pl.Boolean,
    "death_date_candidates": pl.List(pl.Utf8),
    "last_alive_date": pl.Utf8,
    "last_alive_after_death": pl.Boolean,
    "registry_source_ids": pl.List(pl.Utf8),
}


def reduce_vital_status(rows):
    """Reduce parsed registry rows to one record per MRN, independent of row order.

    `rows` are dicts with DFCI_MRN, death_date, last_alive_date (ISO or None)
    and source_id. Conflicting death dates keep the latest (the parent
    pipeline's choice) and are flagged with every candidate retained; a
    last-alive date after the selected death date is flagged, not clamped.
    """
    by_mrn = {}
    for row in rows:
        entry = by_mrn.setdefault(int(row["DFCI_MRN"]), {"death": set(), "alive": set(), "ids": set()})
        if row.get("death_date"):
            entry["death"].add(row["death_date"])
        if row.get("last_alive_date"):
            entry["alive"].add(row["last_alive_date"])
        if row.get("source_id"):
            entry["ids"].add(row["source_id"])
    out = []
    for mrn in sorted(by_mrn):
        entry = by_mrn[mrn]
        deaths = sorted(entry["death"])
        death = deaths[-1] if deaths else None
        alive = max(entry["alive"]) if entry["alive"] else None
        out.append({
            "DFCI_MRN": mrn,
            "death_date": death,
            "death_date_conflict": len(deaths) > 1,
            "death_date_candidates": deaths,
            "last_alive_date": alive,
            "last_alive_after_death": bool(death and alive and alive > death),
            "registry_source_ids": sorted(entry["ids"]),
        })
    return out


def load_vital_status(mrns=None, path=None):
    """One row per MRN from the registry: death and last-alive dates with conflict flags."""
    path = path or source_path("registry")
    lazy = _scan(path, mrns)
    columns = set(lazy.collect_schema().names())
    missing = {"DFCI_MRN", "HYBRID_DEATH_DT", "DERIVED_LAST_ALIVE_DATE"} - columns
    if missing:
        raise ValueError(f"{path} is missing registry columns: {sorted(missing)}")
    frame = lazy.select(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        pl.col("source_row").cast(pl.Int64),
        pl.col("HYBRID_DEATH_DT").cast(pl.Utf8).alias("death"),
        pl.col("DERIVED_LAST_ALIVE_DATE").cast(pl.Utf8).alias("alive"),
    ).collect()
    # Parse before reducing so an unparseable value never wins a max().
    parsed = [
        {
            "DFCI_MRN": row["DFCI_MRN"],
            "death_date": to_iso_date(row["death"]),
            "last_alive_date": to_iso_date(row["alive"]),
            "source_id": f"registry:{row['source_row']}",
        }
        for row in frame.iter_rows(named=True)
        if row["DFCI_MRN"] is not None
    ]
    rows = reduce_vital_status(parsed)
    return _frame(rows, VITAL_STATUS_SCHEMA)
