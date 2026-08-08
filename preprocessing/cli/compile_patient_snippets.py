"""Compile and persist the patient snippets consumed by the NEPC classifier."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.bundles.snippet_bundle import (  # noqa: E402
    SNIPPET_BUNDLE_FILENAME,
    write_snippet_bundle,
)
from preprocessing.config import DEFAULT_OUTPUT_DIR, DEFAULT_PROFILE_NOTE_PATHS, SNIPPET_PROFILES  # noqa: E402
from preprocessing.notes import (  # noqa: E402
    load_notes,
    load_profile_note_mrns,
    load_selected_mrns,
    resolve_note_source,
)
from preprocessing.snippets import build_patient_snippets  # noqa: E402
from preprocessing.triggers import TRIGGER_REGEX, combined_text_pattern  # noqa: E402

_PROFILE = SNIPPET_PROFILES["binary_nepc"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean and trigger-scan prostate notes, rank patient snippets, and "
            "save the standalone artifact required by tasks/binary_NEPC/run_NEPC_classifier.py."
        )
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--notes-parquet", type=Path, action="append", default=None,
                        help="PROFILE_DATA clinical-note parquet. Repeat for multiple files; "
                             "defaults to pathology, imaging, and progress notes.")
    parser.add_argument(
        "--note-bundle-path",
        type=Path,
        default=None,
        help="Standardized Parquet note bundle override.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / SNIPPET_BUNDLE_FILENAME,
    )
    parser.add_argument("--max-notes-per-patient", type=int, default=75)
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=None,
        help="Processes for note cleaning and trigger scanning (default: all cores).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing patient snippet bundle.",
    )
    return parser.parse_args()


def run(args):
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Patient snippet bundle already exists: {args.output_path}. "
            "Use --overwrite to rebuild it."
        )

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    parquet_paths = (
        args.notes_parquet or DEFAULT_PROFILE_NOTE_PATHS
        if args.note_bundle_path is None
        else None
    )
    if parquet_paths is not None and selected_mrns is None:
        raise ValueError(
            "Binary NEPC is prostate-specific. Direct PROFILE_DATA parquet runs "
            "require --mrns or --mrn-file to define the prostate cohort."
        )
    source_label, source_path = resolve_note_source(
        parquet_paths=parquet_paths,
        bundle_path=args.note_bundle_path,
    )
    print(f"Note source: {source_label} ({source_path})")

    notes_df = load_notes(
        parquet_paths=parquet_paths,
        bundle_path=args.note_bundle_path,
        selected_mrns=selected_mrns,
        text_pattern=combined_text_pattern(TRIGGER_REGEX),
    )
    patients_with_notes = (
        load_profile_note_mrns(parquet_paths, selected_mrns)
        if parquet_paths is not None
        else {int(mrn) for mrn in notes_df["DFCI_MRN"].unique().drop_nulls().to_list()}
    )
    all_mrns = set(selected_mrns) if selected_mrns is not None else patients_with_notes
    no_note_mrns = all_mrns - patients_with_notes
    print(f"Loaded notes: {len(notes_df)} rows for {len(all_mrns)} patients")

    patient_snippets = build_patient_snippets(
        notes_df,
        max_notes_per_patient=args.max_notes_per_patient,
        context_chars=_PROFILE.context_chars,
        snippet_max_chars=_PROFILE.max_chars,
        payload_max_chars=_PROFILE.payload_max_chars,
        max_workers=args.scan_workers,
    )
    write_snippet_bundle(
        args.output_path,
        all_mrns=all_mrns,
        patient_snippets=patient_snippets,
        metadata={
            "note_source": source_label,
            "note_source_path": None if source_path is None else str(source_path),
            "max_notes_per_patient": args.max_notes_per_patient,
            "scan_workers": args.scan_workers,
            "patients_with_notes": sorted(patients_with_notes),
            "no_note_mrns": sorted(no_note_mrns),
        },
    )

    print(f"Wrote patient snippet bundle: {args.output_path}")
    print(f"Patients with triggered snippets: {len(patient_snippets)}")
    print(f"Patients with no signal: {len(all_mrns - set(patient_snippets))}")
    print(f"Patients with no notes: {len(no_note_mrns)}")
    print(f"Total saved snippets: {sum(map(len, patient_snippets.values()))}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
