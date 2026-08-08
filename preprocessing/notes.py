"""MRN parsing and note loading from PROFILE_DATA or derived Parquet artifacts."""

import math
import re
from datetime import datetime
from pathlib import Path

import polars as pl
from dateutil import parser as date_parser

from preprocessing.config import (
    DEFAULT_PROFILE_NOTE_PATHS,
    NOTE_BUNDLE_COLUMNS,
    PROFILE_PATH_IMAGE_COLUMNS,
    PROFILE_PROGRESS_COLUMNS,
)
from preprocessing.parquet_io import write_parquet_atomic


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_numeric_scalar(value):
    """Best-effort scalar -> float, returning None on failure (pd.to_numeric(errors='coerce') scalar analogue)."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# MRN parsing
def parse_mrn_values(values):
    mrns = set()
    for value in values:
        if _is_missing(value):
            continue
        for token in re.split(r"[\s,|]+", str(value).strip()):
            if not token:
                continue
            mrn = _to_numeric_scalar(token)
            if mrn is not None:
                mrns.add(int(mrn))
    return mrns


def load_selected_mrns(mrns_arg=None, mrn_file=None):
    selected = set()
    if mrns_arg:
        selected.update(parse_mrn_values([mrns_arg]))
    if mrn_file:
        mrn_file = Path(mrn_file)
        suffix = mrn_file.suffix.lower()
        if suffix == ".csv":
            mrn_df = pl.read_csv(mrn_file, infer_schema_length=None)
        elif suffix == ".parquet":
            mrn_df = pl.read_parquet(mrn_file)
        else:
            raise ValueError(f"MRN cohort file must be CSV or Parquet: {mrn_file}")
        if "DFCI_MRN" in mrn_df.columns:
            selected.update(parse_mrn_values(mrn_df["DFCI_MRN"].to_list()))
        elif mrn_df.height > 0:
            selected.update(parse_mrn_values(mrn_df[:, 0].to_list()))
    return selected or None


def normalize_mrn_column(df):
    if df.is_empty() or "DFCI_MRN" not in df.columns:
        return df
    work = df.with_columns(
        pl.col("DFCI_MRN").cast(pl.Float64, strict=False).alias("DFCI_MRN")
    )
    work = work.drop_nulls(subset=["DFCI_MRN"])
    work = work.with_columns(pl.col("DFCI_MRN").cast(pl.Int64))
    return work


# Note text utilities
def basic_clean_text(text):
    cleaned = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", " ")
        .replace("\xa0", " ")
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def deduplicate_texts(text_entries):
    seen = set()
    deduped = []
    for entry in text_entries:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def to_iso_date(value):
    """Parse a scalar date-like value to an ISO 'YYYY-MM-DD' string, or None.

    Uses dateutil (not a columnar polars op) because this is called per-scalar,
    often millions of times across snippet building; batching isn't applicable here.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (datetime,)):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = date_parser.parse(text)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.strftime("%Y-%m-%d")


def _standardize_note_df(note_df):
    """Select bundle columns, normalize EVENT_DATE to ISO strings, sort deterministically."""
    if note_df.is_empty():
        return pl.DataFrame(schema={c: pl.Utf8 for c in NOTE_BUNDLE_COLUMNS})
    keep_cols = [c for c in NOTE_BUNDLE_COLUMNS if c in note_df.columns]
    standardized = note_df.select(keep_cols)
    if "EVENT_DATE" in standardized.columns:
        # PROFILE EVENT_DATE values are ISO datetime strings that may carry a
        # timezone offset (e.g. "2021-03-14T00:00:00-05:00"). Polars refuses to infer a
        # format when a timezone is present, and we only want the calendar date anyway,
        # so extract the leading YYYY-MM-DD directly rather than parsing to a datetime.
        standardized = standardized.with_columns(
            pl.col("EVENT_DATE")
            .cast(pl.Utf8)
            .str.extract(r"^(\d{4}-\d{2}-\d{2})", 1)
            .alias("EVENT_DATE")
        )
    standardized = standardized.sort(
        ["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], nulls_last=True
    )
    return standardized


def write_note_bundle(path, note_df, *, selected_mrns=None):
    """Write standardized notes as a Parquet bundle."""
    standardized = _standardize_note_df(note_df)
    write_parquet_atomic(standardized, path)


def write_notes_parquet(path, note_df):
    """Write standardized note rows to a Parquet artifact."""
    standardized = _standardize_note_df(note_df)
    write_parquet_atomic(standardized, path)
    return standardized


def load_note_bundle(path, selected_mrns=None):
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Note bundle not found: {bundle_path}")
    try:
        df = normalize_mrn_column(pl.read_parquet(bundle_path))
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Invalid note Parquet bundle: {bundle_path}") from exc
    if df.is_empty():
        raise ValueError(f"No note rows in bundle: {bundle_path}")
    if selected_mrns is not None:
        df = df.filter(pl.col("DFCI_MRN").is_in(selected_mrns))
        if df.is_empty():
            raise ValueError("No notes after MRN filter.")
    return df


def _profile_note_type(path):
    """Map a PROFILE_DATA clinical-note parquet basename to our note taxonomy."""
    name = Path(path).stem.upper()
    if name == "PATHOLOGY_NOTES":
        return "Pathology"
    if name == "IMAGING_NOTES":
        return "Imaging"
    if name == "PROGRESS_NOTES":
        return "Clinician"
    raise ValueError(
        f"Unsupported PROFILE_DATA note parquet: {path}. Expected one of "
        "PATHOLOGY_NOTES.parquet, IMAGING_NOTES.parquet, or PROGRESS_NOTES.parquet."
    )


def load_profile_note_mrns(parquet_paths=None, selected_mrns=None):
    """Return patients with at least one note without reading any text columns."""
    paths = [Path(path) for path in (parquet_paths or DEFAULT_PROFILE_NOTE_PATHS)]
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"PROFILE_DATA note parquet not found: {path}")
        lazy = pl.scan_parquet(path).select(
            pl.col("DFCI_MRN").cast(pl.Int64, strict=False)
        )
        if selected_mrns is not None:
            lazy = lazy.filter(pl.col("DFCI_MRN").is_in(selected_mrns))
        frames.append(lazy)
    if not frames:
        return set()
    mrns = (
        pl.concat(frames, how="vertical_relaxed")
        .drop_nulls()
        .unique()
        .collect()["DFCI_MRN"]
        .to_list()
    )
    return {int(mrn) for mrn in mrns}


def load_profile_notes(
    parquet_paths=None, selected_mrns=None, text_pattern=None, note_types=None
):
    """Load and standardize the merged PROFILE_DATA clinical-note parquets.

    The three source files have slightly different metadata columns. They are
    emitted by PROFILE_data_processing with NARRATIVE_TEXT already merged into
    RPT_TEXT for pathology/imaging. This loader maps that native schema to the
    common note shape consumed by every annotation pipeline. Optional MRN and
    regex predicates are applied lazily before materialization.
    """
    paths = [Path(path) for path in (parquet_paths or DEFAULT_PROFILE_NOTE_PATHS)]
    if not paths:
        raise ValueError("At least one PROFILE_DATA note parquet is required.")

    wanted_note_types = (
        {str(value).strip().lower() for value in note_types} if note_types else None
    )
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"PROFILE_DATA note parquet not found: {path}")
        note_type = _profile_note_type(path)
        if wanted_note_types is not None and note_type.lower() not in wanted_note_types:
            continue
        lazy = pl.scan_parquet(path)
        columns = set(lazy.collect_schema().names())
        expected = (
            PROFILE_PROGRESS_COLUMNS
            if note_type == "Clinician"
            else PROFILE_PATH_IMAGE_COLUMNS
        )
        required = set(expected)
        missing = required - columns
        if missing:
            raise ValueError(
                f"PROFILE_DATA note parquet {path} is missing columns: {sorted(missing)}"
            )
        if selected_mrns is not None:
            lazy = lazy.filter(pl.col("DFCI_MRN").cast(pl.Int64, strict=False).is_in(selected_mrns))

        if text_pattern:
            lazy = lazy.filter(
                pl.col("RPT_TEXT").cast(pl.Utf8).str.contains(text_pattern)
            )
        clinical_text = pl.col("RPT_TEXT").fill_null("").str.strip_chars()

        def source_col(name, fallback_name=None):
            if name in columns:
                return pl.col(name).cast(pl.Utf8)
            if fallback_name in columns:
                return pl.col(fallback_name).cast(pl.Utf8)
            return pl.lit(None, dtype=pl.Utf8)

        frames.append(
            lazy.select(
                pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
                pl.col("EVENT_DATE").cast(pl.Utf8),
                pl.lit(note_type).alias("NOTE_TYPE"),
                clinical_text.alias("CLINICAL_TEXT"),
                source_col("FILE").alias("RAW_SOURCE_FILE"),
                source_col("RPT_ID").alias("RAW_NOTE_ID"),
                pl.col("EVENT_DATE").cast(pl.Utf8).alias("RPT_DATE"),
                source_col("RPT_TYPE", "INP_RPT_TYPE").alias("RPT_TYPE"),
                pl.lit(None, dtype=pl.Utf8).alias("SOURCE_STR"),
                source_col("PROC_DESC").alias("PROC_DESC_STR"),
                source_col("PROVIDER_TYPE").alias("PROVIDER_TYPE_STR"),
                source_col("ENCOUNTER_TYPE_DESC").alias("ENCOUNTER_TYPE_DESC_STR"),
            ).filter(pl.col("DFCI_MRN").is_not_null() & (pl.col("CLINICAL_TEXT") != ""))
        )

    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in NOTE_BUNDLE_COLUMNS})
    df = pl.concat(frames, how="vertical_relaxed").collect()
    df = normalize_mrn_column(df)
    if df.is_empty() and text_pattern is None:
        raise ValueError("No note rows found in the selected PROFILE_DATA parquets.")
    return _standardize_note_df(df)


def resolve_note_source(*, parquet_paths=None, bundle_path=None):
    """Return (source_label, source_path) for the note source load_notes would pick.

    Mirrors load_notes' precedence without reading the data, so callers can log
    which source is in use before paying to load it.
    """
    if parquet_paths:
        return "profile_parquet", tuple(Path(path) for path in parquet_paths)
    if bundle_path is not None:
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Explicit note bundle not found: {bundle_path}")
        return "bundle", bundle_path
    return "profile_parquet", DEFAULT_PROFILE_NOTE_PATHS


def load_notes(
    *, parquet_paths=None, bundle_path=None, selected_mrns=None,
    text_pattern=None, note_types=None
):
    """Load clinical notes for the LLM pipelines.

    Precedence: explicit PROFILE parquet > standardized Parquet bundle > the
    three default PROFILE_DATA parquets.
    """
    if parquet_paths:
        return load_profile_notes(
            parquet_paths, selected_mrns, text_pattern=text_pattern,
            note_types=note_types,
        )
    if bundle_path is not None:
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Explicit note bundle not found: {bundle_path}")
        notes = load_note_bundle(bundle_path, selected_mrns)
    else:
        return load_profile_notes(
            DEFAULT_PROFILE_NOTE_PATHS, selected_mrns, text_pattern,
            note_types=note_types,
        )
    if note_types:
        wanted = {str(value).strip().lower() for value in note_types}
        notes = notes.filter(
            pl.col("NOTE_TYPE").cast(pl.Utf8).str.to_lowercase().is_in(wanted)
        )
    return notes
