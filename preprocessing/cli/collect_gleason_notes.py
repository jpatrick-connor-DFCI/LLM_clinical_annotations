"""Stage 1 — Collect notes mentioning a Gleason score / Grade Group / ISUP grade.

Scans notes for every prostate patient, groups matches into per-patient,
payload-sized chunks, and writes them as evidence for the LLM step.

Outputs (under <output-dir>):
  gleason_evidence.tsv        one row per snippet, grouped by patient/chunk
  gleason_evidence.meta.json  scan_config hash + resolved params the evidence
                               was built under (used by build_gleason_timeline.py
                               to validate that chunk-index resume is safe)

Re-running without --overwrite: if the evidence file's recorded scan_config
matches the resolved settings for this invocation, the scan is skipped
entirely (existing evidence is reused as-is). If it differs, this raises —
regenerating evidence with different scan parameters changes chunk_index
assignment, which would silently corrupt stage-2's chunk-level resume; pass
--overwrite to intentionally rescan and discard the old evidence + chunk state.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_DATA_PATH, PROSTATE_TEXT_CSV, SNIPPET_PROFILES  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    evidence_scan_config_key,
    filter_note_types,
    group_patient_snippets,
    read_scan_config_meta,
    write_scan_config_meta,
)
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescan from scratch even if existing evidence matches these settings.",
    )
    return parser.parse_args()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "gleason_evidence.tsv"
    meta_path = args.output_dir / "gleason_evidence.meta.json"
    snippet_max_chars = SNIPPET_PROFILES["longitudinal"].max_chars

    if args.overwrite:
        evidence_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

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

    scan_config = evidence_scan_config_key(
        notes_df,
        TRIGGER_REGEX,
        context_chars=args.context_chars,
        snippet_max_chars=snippet_max_chars,
        payload_max_chars=args.payload_max_chars,
        note_types=args.note_types,
    )

    if evidence_path.exists() and evidence_path.stat().st_size > 0:
        existing_meta = read_scan_config_meta(meta_path)
        if existing_meta is None:
            raise ValueError(
                f"Existing evidence has no scan-config sidecar: {meta_path}. "
                "Re-run with --overwrite to rebuild it safely."
            )
        if existing_meta.get("scan_config") != scan_config:
            raise ValueError(
                "Gleason scan settings differ from the existing evidence "
                f"({existing_meta.get('scan_config')} != {scan_config}). "
                "Re-run with --overwrite instead of mixing incompatible evidence "
                "and chunk state."
            )
        print(f"Existing evidence matches current scan settings, reusing: {evidence_path}")
        return

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
    write_scan_config_meta(
        meta_path,
        scan_config,
        context_chars=args.context_chars,
        snippet_max_chars=snippet_max_chars,
        payload_max_chars=args.payload_max_chars,
        note_types=args.note_types,
    )
    print(f"Wrote Gleason evidence ({evidence.height} rows): {evidence_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
