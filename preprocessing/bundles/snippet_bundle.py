"""Persistence for the binary NEPC patient-snippet Parquet artifact."""

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from preprocessing.parquet_io import write_parquet_atomic


SNIPPET_BUNDLE_FILENAME = "LLM_NEPC_classifier_patient_snippets.parquet"
SNIPPET_BUNDLE_FORMAT = "binary_nepc_patient_snippets"
# Version 3 defines the flat cohort/snippet Parquet contract.
SNIPPET_BUNDLE_VERSION = 3
REQUIRED_SNIPPET_FIELDS = {
    "note_date",
    "note_type",
    "trigger_categories",
    "snippet",
}
BUNDLE_COLUMNS = [
    "bundle_format",
    "bundle_version",
    "created_at",
    "metadata_json",
    "DFCI_MRN",
    "cohort_status",
    "snippet_index",
    "note_date",
    "note_type",
    "trigger_categories",
    "snippet",
]


def _normalize_snippets(patient_snippets):
    normalized = {}
    for raw_mrn, snippets in patient_snippets.items():
        mrn = int(raw_mrn)
        if not isinstance(snippets, list):
            raise ValueError(f"Snippets for MRN {mrn} must be a list")
        normalized_snippets = []
        for index, snippet in enumerate(snippets):
            if not isinstance(snippet, dict):
                raise ValueError(f"Snippet {index} for MRN {mrn} must be an object")
            missing = REQUIRED_SNIPPET_FIELDS - set(snippet)
            if missing:
                raise ValueError(
                    f"Snippet {index} for MRN {mrn} is missing: {sorted(missing)}"
                )
            if not isinstance(snippet["trigger_categories"], list):
                raise ValueError(
                    f"trigger_categories for snippet {index}, MRN {mrn} must be a list"
                )
            if not isinstance(snippet["snippet"], str):
                raise ValueError(
                    f"snippet text for snippet {index}, MRN {mrn} must be a string"
                )
            normalized_snippets.append(
                {
                    "note_date": snippet["note_date"],
                    "note_type": snippet["note_type"],
                    "trigger_categories": list(snippet["trigger_categories"]),
                    "snippet": snippet["snippet"],
                }
            )
        normalized[mrn] = normalized_snippets
    return normalized


def _empty_bundle_frame():
    return pl.DataFrame(
        schema={
            "bundle_format": pl.String,
            "bundle_version": pl.Int64,
            "created_at": pl.String,
            "metadata_json": pl.String,
            "DFCI_MRN": pl.Int64,
            "cohort_status": pl.String,
            "snippet_index": pl.Int64,
            "note_date": pl.String,
            "note_type": pl.String,
            "trigger_categories": pl.List(pl.String),
            "snippet": pl.String,
        }
    )


def write_snippet_bundle(path, *, all_mrns, patient_snippets, metadata=None):
    """Atomically write one flat Parquet row set covering the entire cohort."""
    path = Path(path)
    all_mrns = sorted({int(mrn) for mrn in all_mrns})
    normalized = _normalize_snippets(patient_snippets)
    unexpected = set(normalized) - set(all_mrns)
    if unexpected:
        raise ValueError(
            f"Snippet patients absent from all_mrns: {sorted(unexpected)[:10]}"
        )

    metadata = metadata or {}
    no_note_mrns = {int(mrn) for mrn in metadata.get("no_note_mrns", [])}
    common = {
        "bundle_format": SNIPPET_BUNDLE_FORMAT,
        "bundle_version": SNIPPET_BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }
    rows = []
    for mrn in all_mrns:
        snippets = normalized.get(mrn, [])
        if snippets:
            for index, snippet in enumerate(snippets):
                rows.append(
                    {
                        **common,
                        "DFCI_MRN": mrn,
                        "cohort_status": "triggered",
                        "snippet_index": index,
                        **snippet,
                    }
                )
        else:
            rows.append(
                {
                    **common,
                    "DFCI_MRN": mrn,
                    "cohort_status": (
                        "no_notes" if mrn in no_note_mrns else "no_trigger"
                    ),
                    "snippet_index": None,
                    "note_date": None,
                    "note_type": None,
                    "trigger_categories": None,
                    "snippet": None,
                }
            )
    frame = (
        pl.DataFrame({column: [row.get(column) for row in rows] for column in BUNDLE_COLUMNS})
        if rows
        else _empty_bundle_frame()
    )
    write_parquet_atomic(frame, path)


def load_snippet_bundle(path, *, selected_mrns=None):
    """Load and validate a snippet Parquet artifact, optionally filtering its cohort."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Patient snippet Parquet not found: {path}. "
            "Run preprocessing/cli/compile_patient_snippets.py first."
        )
    try:
        frame = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"Invalid patient snippet Parquet: {path}") from error
    missing = set(BUNDLE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Patient snippet Parquet is missing: {sorted(missing)}")
    if frame.is_empty():
        return set(), {}, {}

    formats = set(frame["bundle_format"].drop_nulls().to_list())
    if formats != {SNIPPET_BUNDLE_FORMAT}:
        raise ValueError(f"Unrecognized patient snippet bundle format: {path}")
    versions = set(frame["bundle_version"].drop_nulls().to_list())
    if versions != {SNIPPET_BUNDLE_VERSION}:
        raise ValueError(
            f"Unsupported patient snippet bundle version {sorted(versions)!r}: {path}"
        )
    try:
        metadata = json.loads(frame["metadata_json"].drop_nulls().item(0))
    except (json.JSONDecodeError, TypeError, IndexError) as error:
        raise ValueError(f"Invalid patient snippet metadata: {path}") from error

    if selected_mrns is not None:
        selected_mrns = {int(mrn) for mrn in selected_mrns}
        frame = frame.filter(pl.col("DFCI_MRN").is_in(selected_mrns))
    all_mrns = {int(mrn) for mrn in frame["DFCI_MRN"].drop_nulls().to_list()}
    patient_snippets = {}
    triggered = frame.filter(pl.col("cohort_status") == "triggered").sort(
        ["DFCI_MRN", "snippet_index"]
    )
    for row in triggered.iter_rows(named=True):
        patient_snippets.setdefault(int(row["DFCI_MRN"]), []).append(
            {
                "note_date": row["note_date"],
                "note_type": row["note_type"],
                "trigger_categories": row["trigger_categories"] or [],
                "snippet": row["snippet"],
            }
        )
    return all_mrns, _normalize_snippets(patient_snippets), metadata
