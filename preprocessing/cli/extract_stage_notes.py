"""Stage 1 — Scan clinical notes for stage mentions and write an evidence table.

For every patient, notes are scanned for stage triggers, context windows are
extracted around each match, copy-forward notes are de-duplicated per patient,
and the resulting snippets are written to a Parquet evidence table. This step runs
before any LLM calls so the scanning layer can be audited and re-used independently.

Default source: all three merged PROFILE_DATA clinical-note parquets, covering the
full pan-cancer population (no pre-specified MRN list required).

Outputs (under <output-dir>):
  stage_evidence.parquet    Deduped snippets — one row per unique (patient, snippet).

Usage:
  python preprocessing/cli/extract_stage_notes.py --output-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_PROFILE_NOTE_PATHS  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    evidence_scan_config_key,
    file_sha256,
    iter_note_snippets,
    read_scan_config_meta,
    write_scan_config_meta,
)
from preprocessing.notes import (  # noqa: E402
    load_note_bundle,
    load_profile_notes,
    load_selected_mrns,
)
from preprocessing.parquet_io import write_parquet_atomic  # noqa: E402
from preprocessing.triggers import combined_text_pattern  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("STAGE_OUTPUT_DIR", "/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/")
)

STAGE_TRIGGER_REGEX = {
    "stage_group": (
        r"\b(?:(?:clinical|pathologic|pathological|overall|ajcc)\s+)?stage\s+"
        r"(?:IV|III|II|I|[1-4]|one|two|three|four)(?:[A-Ca-c]\d?|\d)?\b"
    ),
    "tnm": r"\b[cpyru]?T(?:is|x|[0-4][a-d]?)\s*[,/ ]*N(?:x|[0-3][a-c]?)\s*[,/ ]*M(?:x|[0-1][a-c]?)\b",
    "figo": r"\bFIGO(?:\s+stage)?\s+(?:IV|III|II|I|[1-4])(?:[A-Ca-c]\d?|\d)?\b",
    "ann_arbor": r"\bAnn\s+Arbor(?:\s+stage)?\s+(?:IV|III|II|I)(?:[ABESX]+)?\b",
    "rai_binet": r"\b(?:Rai\s+stage\s+[0-4]|Binet\s+stage\s+[ABC])\b",
    "durie_salmon": r"\b(?:Durie[- ]Salmon\s+)?stage\s+(?:III|II|I)[AB]?\b",
    "limited_extensive": r"\b(?:limited|extensive)[- ]stage\s+(?:small[- ]cell|SCLC)\b",
}

EVIDENCE_COLUMNS = ["note_uid", "DFCI_MRN", "note_date", "note_type", "trigger_categories", "snippet"]


def _records_to_rows(records):
    """Convert snippet dictionaries to flat evidence rows."""
    return [
        {
            "note_uid": r["note_uid"],
            "DFCI_MRN": r["DFCI_MRN"],
            "note_date": r["note_date"],
            "note_type": r["note_type"],
            "trigger_categories": r["trigger_categories"],
            "snippet": r["snippet"],
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# Note loading and trigger scan
# ---------------------------------------------------------------------------

def _load_stage_notes(args, selected_mrns):
    use_profile_parquets = args.notes_parquet is not None or (
        args.note_bundle_path is None
    )
    if use_profile_parquets:
        parquet_paths = args.notes_parquet or list(DEFAULT_PROFILE_NOTE_PATHS)
        notes_df = load_profile_notes(
            parquet_paths,
            selected_mrns,
            text_pattern=combined_text_pattern(STAGE_TRIGGER_REGEX),
            note_types=getattr(args, "note_types", None),
        )
        print(
            f"Loaded PROFILE_DATA parquets: {len(notes_df)} candidate rows for "
            f"{notes_df['DFCI_MRN'].n_unique()} patients"
        )
    elif args.note_bundle_path is not None:
        notes_df = load_note_bundle(args.note_bundle_path, selected_mrns)
        print(f"Loaded bundle: {len(notes_df)} rows for {notes_df['DFCI_MRN'].n_unique()} patients")
    return notes_df, use_profile_parquets


def _load_and_scan_sequential(args, selected_mrns, context_chars):
    """Backward-compatible helper used by focused preprocessing tests."""
    notes_df, _ = _load_stage_notes(args, selected_mrns)
    return list(iter_note_snippets(notes_df, STAGE_TRIGGER_REGEX, context_chars=context_chars))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan clinical notes for stage mentions and write a snippet evidence table. "
                    "By default scans all PROFILE_DATA note parquets pan-cancer."
    )
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="Optional: restrict the scan using a CSV DFCI_MRN cohort.")
    parser.add_argument("--mrns", default=None,
                        help="Optional: comma- or space-separated MRNs to restrict the scan.")
    parser.add_argument("--notes-parquet", type=Path, action="append", default=None,
                        help="PROFILE_DATA clinical-note parquet. Repeat for multiple files; "
                             "defaults to pathology, imaging, and progress notes.")
    parser.add_argument("--note-bundle-path", type=Path, default=None,
                        help="Optional: load notes from a standardized Parquet bundle.")
    parser.add_argument("--note-types", nargs="+", default=None,
                        help="Optional: restrict to these NOTE_TYPE values "
                             "(e.g. Pathology Clinician). Default: all note types.")
    parser.add_argument("--context-chars", type=int, default=600,
                        help="Characters of context on each side of a trigger match (default: 600).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear all existing output files and rescan from scratch.")
    return parser.parse_args()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "stage_evidence.parquet"
    meta_path = args.output_dir / "stage_evidence.meta.parquet"

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    if args.overwrite:
        for path in (
            evidence_path,
            meta_path,
            args.output_dir / "stage_extractions_raw.parquet",
            args.output_dir / "stage_processed_patients.parquet",
            args.output_dir / "stage_timeline.parquet",
            args.output_dir / "stage_run.parquet",
        ):
            path.unlink(missing_ok=True)

    notes_df, direct_parquet = _load_stage_notes(args, selected_mrns)
    if args.note_types and not direct_parquet:
        wanted = {t.strip().lower() for t in args.note_types}
        notes_df = notes_df.filter(
            pl.col("NOTE_TYPE").cast(pl.Utf8).str.to_lowercase().is_in(wanted)
        )
    scan_config = evidence_scan_config_key(
        notes_df,
        STAGE_TRIGGER_REGEX,
        context_chars=args.context_chars,
        snippet_max_chars=30_000,
        payload_max_chars=60_000,
        note_types=args.note_types,
    )
    if evidence_path.exists() and evidence_path.stat().st_size > 0:
        existing = read_scan_config_meta(meta_path)
        if (
            existing
            and existing.get("scan_config") == scan_config
            and existing.get("evidence_sha256") == file_sha256(evidence_path)
        ):
            print(f"Existing stage evidence matches current inputs: {evidence_path}")
            return
        raise ValueError("Stage evidence inputs changed; re-run with --overwrite.")

    records = list(iter_note_snippets(
        notes_df, STAGE_TRIGGER_REGEX, context_chars=args.context_chars
    ))
    if args.note_types and not direct_parquet:
        wanted = {t.strip().lower() for t in args.note_types}
        before = len(records)
        records = [r for r in records if (r["note_type"] or "").lower() in wanted]
        print(f"After note-type filter {args.note_types}: {len(records)}/{before} snippets")
    evidence_rows = _records_to_rows(records)
    if evidence_rows:
        evidence_df = pl.DataFrame(
            {c: [r.get(c) for r in evidence_rows] for c in EVIDENCE_COLUMNS}
        )
        evidence_df = evidence_df.sort(["DFCI_MRN", "note_date"], nulls_last=True)
    else:
        evidence_df = pl.DataFrame(schema={c: pl.Utf8 for c in EVIDENCE_COLUMNS})
    write_parquet_atomic(evidence_df, evidence_path)
    write_scan_config_meta(
        meta_path,
        scan_config,
        context_chars=args.context_chars,
        note_types=args.note_types,
        evidence_sha256=file_sha256(evidence_path),
    )
    n = evidence_df.height

    n_patients = evidence_df["DFCI_MRN"].n_unique() if not evidence_df.is_empty() else 0
    print(f"Wrote {n} evidence snippets for {n_patients} patients: {evidence_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
