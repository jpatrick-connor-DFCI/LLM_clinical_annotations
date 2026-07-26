"""MRN parsing and note loading: raw OncDRS JSON, gzip bundle, or compiled CSV."""

import gzip
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dateutil import parser as date_parser

try:
    import ijson
except ImportError:
    ijson = None

from preprocessing.config import DEFAULT_RAW_TEXT_PATHS, NOTE_BUNDLE_COLUMNS, PROSTATE_TEXT_CSV


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
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            mrn_df = pl.read_csv(mrn_file, separator=sep, infer_schema_length=None)
            if "DFCI_MRN" in mrn_df.columns:
                selected.update(parse_mrn_values(mrn_df["DFCI_MRN"].to_list()))
            elif mrn_df.height > 0:
                selected.update(parse_mrn_values(mrn_df[:, 0].to_list()))
        else:
            with open(mrn_file, "r", encoding="utf-8") as handle:
                selected.update(parse_mrn_values(handle.readlines()))
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


# Raw / bundle loaders
def infer_note_type_from_filename(path):
    name = Path(path).name.lower()
    if "imaging" in name:
        return "Imaging"
    if "prognote" in name or "progress" in name or "clinic" in name:
        return "Clinician"
    if "pathology" in name or re.search(r"(^|[-_])path(?:[-_.]|$)", name):
        return "Pathology"
    return None


def discover_raw_text_files(raw_text_paths):
    discovered = []
    seen = set()
    for raw_text_path in raw_text_paths:
        raw_text_path = Path(raw_text_path)
        if not raw_text_path.exists():
            continue
        for path in sorted(raw_text_path.rglob("*.json")):
            note_type = infer_note_type_from_filename(path)
            if note_type is not None and str(path) not in seen:
                seen.add(str(path))
                discovered.append((path, note_type))
    return discovered


def extract_raw_docs(payload):
    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("docs"), list):
            return response["docs"]
        if isinstance(payload.get("docs"), list):
            return payload["docs"]
    if isinstance(payload, list):
        return payload
    return []


def iter_raw_docs_from_file(path):
    if ijson is None:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from extract_raw_docs(payload)
        return

    with open(path, "r", encoding="utf-8") as handle:
        for prefix in ("response.docs.item", "docs.item", "item"):
            handle.seek(0)
            try:
                iterator = ijson.items(handle, prefix)
                first = next(iterator, None)
            except ijson.JSONError:
                continue
            if first is None:
                continue
            yield first
            yield from iterator
            return


def build_raw_note_row(note, note_type, source_file):
    mrn = _to_numeric_scalar(note.get("DFCI_MRN"))
    if mrn is None:
        return None
    text_entries = [v for k, v in note.items() if "TEXT" in str(k).upper()]
    text = basic_clean_text(" ".join(deduplicate_texts(text_entries)))
    if not text:
        return None
    return {
        "DFCI_MRN": int(mrn),
        "EVENT_DATE": note.get("EVENT_DATE") or note.get("RPT_DATE"),
        "NOTE_TYPE": note_type,
        "CLINICAL_TEXT": text,
        "RAW_SOURCE_FILE": Path(source_file).name,
        "RAW_NOTE_ID": note.get("id"),
        "RPT_DATE": note.get("RPT_DATE"),
        "RPT_TYPE": note.get("RPT_TYPE"),
        "SOURCE_STR": note.get("SOURCE_STR"),
        "PROC_DESC_STR": note.get("PROC_DESC_STR"),
        "ENCOUNTER_TYPE_DESC_STR": note.get("ENCOUNTER_TYPE_DESC_STR"),
    }


def resolve_raw_text_paths(raw_text_paths_arg=None):
    if not raw_text_paths_arg:
        return list(DEFAULT_RAW_TEXT_PATHS)
    seen, paths = set(), []
    for path in raw_text_paths_arg:
        normalized = Path(path)
        key = str(normalized)
        if key not in seen:
            seen.add(key)
            paths.append(normalized)
    return paths


def load_raw_text_notes(raw_text_paths, selected_mrns):
    if selected_mrns is None:
        raise ValueError("Raw text mode requires --mrns or --mrn-file.")
    raw_files = discover_raw_text_files(raw_text_paths)
    if not raw_files:
        joined = ", ".join(str(p) for p in raw_text_paths)
        raise FileNotFoundError(f"No supported raw JSON note files found under: {joined}")
    rows = []
    for file_path, note_type in raw_files:
        for note in iter_raw_docs_from_file(file_path):
            mrn = _to_numeric_scalar(note.get("DFCI_MRN"))
            if mrn is None or int(mrn) not in selected_mrns:
                continue
            row = build_raw_note_row(note, note_type, file_path)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError("No raw notes were found for the requested MRNs.")
    df = normalize_mrn_column(pl.DataFrame(rows, infer_schema_length=None))
    if df.is_empty():
        raise ValueError("No raw notes were found for the requested MRNs.")
    return df


def _standardize_note_df(note_df):
    """Select bundle columns, normalize EVENT_DATE to ISO strings, sort deterministically."""
    if note_df.is_empty():
        return pl.DataFrame(schema={c: pl.Utf8 for c in NOTE_BUNDLE_COLUMNS})
    keep_cols = [c for c in NOTE_BUNDLE_COLUMNS if c in note_df.columns]
    standardized = note_df.select(keep_cols)
    if "EVENT_DATE" in standardized.columns:
        # OncDRS EVENT_DATE/RPT_DATE values are ISO datetime strings that may carry a
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


def write_note_bundle(path, note_df, *, raw_text_paths=None, selected_mrns=None):
    standardized = _standardize_note_df(note_df)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_mrn_count": len(selected_mrns) if selected_mrns is not None else None,
        "patient_count": int(standardized["DFCI_MRN"].n_unique()) if not standardized.is_empty() else 0,
        "note_count": int(standardized.height),
        "raw_text_paths": [str(p) for p in raw_text_paths] if raw_text_paths else None,
        "notes": standardized.to_dicts() if not standardized.is_empty() else [],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def write_notes_csv(path, note_df):
    """Write standardized note rows to a CSV (the default LLM-pipeline note source)."""
    standardized = _standardize_note_df(note_df)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.write_csv(output_path)
    return standardized


def load_note_bundle(path, selected_mrns=None):
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Note bundle not found: {bundle_path}")
    with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get("notes", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    df = normalize_mrn_column(pl.DataFrame(records, infer_schema_length=None) if records else pl.DataFrame())
    if df.is_empty():
        raise ValueError(f"No note rows in bundle: {bundle_path}")
    if selected_mrns is not None:
        df = df.filter(pl.col("DFCI_MRN").is_in(selected_mrns))
        if df.is_empty():
            raise ValueError("No notes after MRN filter.")
    return df


def load_notes_csv(csv_path, selected_mrns=None):
    """Load the compiled prostate notes CSV produced by compile_prostate_notes.py.

    Uses a lazy scan so the MRN filter (when given) is pushed down during the
    multi-threaded CSV read instead of materializing the full file first.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prostate notes CSV not found: {csv_path}")
    # Read every known column as Utf8 rather than scanning the whole file to infer
    # types (infer_schema_length=None). The snippet builder treats all fields as
    # strings anyway, and DFCI_MRN is normalized numerically downstream, so a fixed
    # Utf8 schema is both faster to load and safe. Unknown columns still infer.
    lazy = pl.scan_csv(
        csv_path,
        schema_overrides={c: pl.Utf8 for c in NOTE_BUNDLE_COLUMNS},
        infer_schema_length=1000,
    )
    if "CLINICAL_TEXT" not in lazy.collect_schema().names():
        raise ValueError(f"Prostate notes CSV missing CLINICAL_TEXT column: {csv_path}")
    if selected_mrns is not None:
        lazy = lazy.filter(
            pl.col("DFCI_MRN").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).is_in(selected_mrns)
        )
    df = normalize_mrn_column(lazy.collect())
    if df.is_empty():
        raise ValueError(f"No note rows in CSV: {csv_path}")
    return df


def resolve_note_source(*, csv_path=None, bundle_path=None):
    """Return (source_label, source_path) for the note source load_notes would pick.

    Mirrors load_notes' precedence without reading the data, so callers can log
    which source is in use before paying to load it. The raw-JSON path is much
    slower than the CSV/bundle, so surfacing an accidental fall-through matters.
    """
    if bundle_path is not None and Path(bundle_path).exists():
        return "bundle", Path(bundle_path)
    csv_path = csv_path or PROSTATE_TEXT_CSV
    if Path(csv_path).exists():
        return "csv", Path(csv_path)
    return "raw_json", None


def load_notes(*, csv_path=None, bundle_path=None, raw_text_paths=None, selected_mrns=None):
    """Load prostate notes for the LLM pipelines.

    Precedence: an explicitly-provided bundle that exists > the compiled
    prostate_text_data.csv > raw OncDRS JSONs. The CSV is the default source.
    """
    if bundle_path is not None and Path(bundle_path).exists():
        return load_note_bundle(bundle_path, selected_mrns)
    csv_path = csv_path or PROSTATE_TEXT_CSV
    if Path(csv_path).exists():
        return load_notes_csv(csv_path, selected_mrns)
    return load_raw_text_notes(resolve_raw_text_paths(raw_text_paths), selected_mrns)
