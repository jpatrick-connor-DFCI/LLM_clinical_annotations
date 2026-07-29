"""Stage 1 — Collect notes mentioning a Gleason score / Grade Group / ISUP grade.

Scans notes for every prostate patient, groups matches into per-patient,
payload-sized chunks, and writes them as evidence for the LLM step.

Outputs (under <output-dir>):
  gleason_evidence.tsv   one row per snippet, grouped by patient/chunk
"""

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_DATA_PATH, PROSTATE_TEXT_CSV, SNIPPET_PROFILES  # noqa: E402
from preprocessing.longitudinal import filter_note_types, group_patient_snippets  # noqa: E402
from preprocessing.notes import load_notes, load_selected_mrns  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_gleason_timeline"
_PROFILE = SNIPPET_PROFILES["longitudinal"]

# Any mention of Gleason / Grade Group / ISUP grading collects the note.
TRIGGER_REGEX = {
    "gleason": r"\b(?:gleason|grade\s+group|isup(?:\s+grade)?)\b",
}

EVIDENCE_COLUMNS = ["DFCI_MRN", "chunk_index", "note_date", "note_type", "snippet"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect notes mentioning a Gleason score / Grade Group / ISUP grade."
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--notes-csv", type=Path, default=PROSTATE_TEXT_CSV)
    parser.add_argument("--note-bundle-path", type=Path, default=None)
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None)
    parser.add_argument(
        "--note-types",
        nargs="+",
        default=None,
        help="Restrict to these NOTE_TYPE values (e.g. Pathology). Default: all notes. "
        "Gleason is authoritatively assigned in pathology, so 'Pathology' is far "
        "cheaper and higher-fidelity.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=600,
        help="Chars of context kept on each side of a Gleason match. Smaller windows "
        "raise the copy-forward dedup hit-rate and pack more notes per call.",
    )
    parser.add_argument(
        "--payload-max-chars",
        type=int,
        default=_PROFILE.payload_max_chars,
        help="Max snippet chars packed into one LLM call (one chunk per patient until full).",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=None,
        help="Processes for note cleaning and trigger scanning (default: all cores).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "gleason_evidence.tsv"

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    notes_df = load_notes(
        csv_path=args.notes_csv,
        bundle_path=args.note_bundle_path,
        raw_text_paths=args.raw_text_path,
        selected_mrns=selected_mrns,
    )
    print(
        f"Loaded notes: {len(notes_df)} rows for "
        f"{notes_df['DFCI_MRN'].n_unique()} patients"
    )

    if args.note_types:
        notes_df = filter_note_types(notes_df, args.note_types)
        print(f"After note-type filter {args.note_types}: {len(notes_df)} rows")

    patient_chunks = group_patient_snippets(
        notes_df,
        TRIGGER_REGEX,
        context_chars=args.context_chars,
        payload_max_chars=args.payload_max_chars,
        max_workers=args.scan_workers,
    )
    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients mentioning Gleason: {len(patient_chunks)} "
        f"({total_chunks} chunks)"
    )

    rows = []
    for mrn, chunks in patient_chunks.items():
        for chunk_index, chunk in enumerate(chunks):
            for rec in chunk:
                rows.append({
                    "DFCI_MRN": int(mrn),
                    "chunk_index": chunk_index,
                    "note_date": rec["note_date"],
                    "note_type": rec["note_type"],
                    "snippet": rec["snippet"],
                })

    if rows:
        evidence = pl.DataFrame({c: [r.get(c) for r in rows] for c in EVIDENCE_COLUMNS})
    else:
        evidence = pl.DataFrame(schema={c: pl.Utf8 for c in EVIDENCE_COLUMNS})
    evidence.write_csv(evidence_path, separator="\t")
    print(f"Wrote Gleason evidence ({evidence.height} rows): {evidence_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
