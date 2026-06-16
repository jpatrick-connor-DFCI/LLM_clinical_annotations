"""Stage 1 — Scan clinical notes for stage mentions and write an evidence table.

For every patient, notes are scanned for stage triggers, context windows are
extracted around each match, copy-forward notes are de-duplicated per patient,
and the resulting snippets are written to a TSV evidence table. This step runs
before any LLM calls so the scanning layer can be audited and re-used independently.

Output (under <output-dir>):
  stage_evidence.tsv    One row per unique (patient, snippet), sorted by patient + date.

Usage:
  python cancer_stage/extract_stage_notes.py [options]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.longitudinal_helpers import (  # noqa: E402
    DEFAULT_DATA_PATH,
    PROSTATE_TEXT_CSV,
    filter_note_types,
    iter_note_snippets,
    load_notes,
    load_selected_mrns,
)
from shared.llm_helpers import (  # noqa: E402
    build_raw_note_row,
    discover_raw_text_files,
    extract_raw_docs,
    normalize_mrn_column,
)

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_stage_timeline"

# Four trigger categories, passed as a dict to find_matches / iter_note_snippets.
# False positives are acceptable — the LLM disambiguates true staging from
# incidental mentions. trigger_categories is recorded per snippet so the model
# knows which evidence type triggered the scan.
STAGE_TRIGGER_REGEX = {
    "stage_group": (
        # "clinical stage IV", "pathologic stage IIIA", "Stage 2b", "stage four"
        r"\b(?:clinical|pathologic|pathological)\s+stage\s+"
        r"(?:IV[ABCabc]?|III[ABCabc]?|II[ABCabc]?|I[ABCabc]?|[0-4][ABCabc]?)\b"
        r"|\bstage\s+"
        r"(?:IV[ABCabc]?|III[ABCabc]?|II[ABCabc]?|I[ABCabc]?|[0-4][ABCabc]?)\b"
        r"|\bstage\s+(?:one|two|three|four)\b"
    ),
    "staging_system": (
        # Any AJCC/FIGO/Ann Arbor mention implies a staging discussion.
        r"\b(?:AJCC|FIGO|Ann\s+Arbor)\b"
    ),
    "tnm": (
        # Require TxNx or full TxNxMx to avoid bare "T2" imaging descriptors.
        # M1 alone (metastatic classification) is included as a standalone trigger.
        r"\b[cpyr]{0,2}[Tt][0-4][a-z]?\s*[Nn][0-3xX]\s*[Mm][01][a-z]?\b"
        r"|\b[cpyr]{0,2}[Tt][0-4][a-z]?\s*[Nn][0-3xX]\b"
        r"|\b[Mm]1[a-z]?\b"
    ),
    "limited_extensive": (
        r"\b(?:limited|extensive)\s+stage\b"
    ),
}

EVIDENCE_COLUMNS = [
    "note_uid",
    "DFCI_MRN",
    "note_date",
    "note_type",
    "trigger_categories",
    "snippet",
]


def _load_all_raw_notes(raw_text_paths):
    """Scan all raw OncDRS JSON files without requiring an MRN filter.

    load_notes() / load_raw_text_notes() rejects a None selected_mrns argument
    because those helpers are designed for cohort-scoped pipelines. The stage
    scan is cancer-agnostic and must be able to sweep the full corpus.
    """
    raw_files = discover_raw_text_files(raw_text_paths)
    if not raw_files:
        joined = ", ".join(str(p) for p in raw_text_paths)
        raise FileNotFoundError(f"No supported raw JSON files found under: {joined}")
    rows = []
    for file_path, note_type in tqdm(raw_files, desc="Scanning raw files", unit="file"):
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for note in extract_raw_docs(payload):
            row = build_raw_note_row(note, note_type, file_path)
            if row is not None:
                rows.append(row)
    return normalize_mrn_column(pd.DataFrame(rows))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan clinical notes for stage mentions and write a snippet evidence table."
    )
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="Optional: restrict scan to these MRNs "
                             "(file of MRNs or CSV with DFCI_MRN column). "
                             "Default: scan all patients.")
    parser.add_argument("--mrns", default=None,
                        help="Optional: comma- or space-separated MRNs to restrict the scan. "
                             "Default: scan all patients.")
    parser.add_argument("--notes-csv", type=Path, default=PROSTATE_TEXT_CSV,
                        help="Pre-compiled prostate notes CSV (default source).")
    parser.add_argument("--note-bundle-path", type=Path, default=None,
                        help="Note bundle (.json.gz) produced by write_note_bundle().")
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None,
                        help="Raw OncDRS JSON directory. Repeat to add multiple paths.")
    parser.add_argument("--note-types", nargs="+", default=None,
                        help="Restrict to these NOTE_TYPE values (e.g. Pathology Clinician). "
                             "Default: all note types.")
    parser.add_argument("--context-chars", type=int, default=2000,
                        help="Characters of context kept on each side of a trigger match. "
                             "Default: 2000.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing stage_evidence.tsv before writing.")
    return parser.parse_args()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "stage_evidence.tsv"

    if args.overwrite and evidence_path.exists():
        evidence_path.unlink()

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    # load_raw_text_notes() requires an MRN filter, so when raw text paths are
    # specified without one we use _load_all_raw_notes() to sweep the full corpus.
    if args.raw_text_path is not None and selected_mrns is None:
        notes_df = _load_all_raw_notes(args.raw_text_path)
    else:
        notes_df = load_notes(
            csv_path=args.notes_csv,
            bundle_path=args.note_bundle_path,
            raw_text_paths=args.raw_text_path,
            selected_mrns=selected_mrns,
        )
    print(
        f"Loaded notes: {len(notes_df)} rows for "
        f"{notes_df['DFCI_MRN'].nunique()} patients"
    )

    if args.note_types:
        notes_df = filter_note_types(notes_df, args.note_types)
        print(f"After note-type filter {args.note_types}: {len(notes_df)} rows")

    rows = []
    for rec in iter_note_snippets(
        notes_df, STAGE_TRIGGER_REGEX, context_chars=args.context_chars
    ):
        rows.append({
            "note_uid": rec["note_uid"],
            "DFCI_MRN": rec["DFCI_MRN"],
            "note_date": rec["note_date"],
            "note_type": rec["note_type"],
            "trigger_categories": ",".join(rec["trigger_categories"]),
            "snippet": rec["snippet"],
        })

    evidence_df = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
    if not evidence_df.empty:
        evidence_df = evidence_df.sort_values(
            ["DFCI_MRN", "note_date"], na_position="last"
        )
    evidence_df.to_csv(evidence_path, sep="\t", index=False)

    n_patients = evidence_df["DFCI_MRN"].nunique() if not evidence_df.empty else 0
    print(
        f"Wrote {len(evidence_df)} evidence snippets for "
        f"{n_patients} patients: {evidence_path}"
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
