"""Compile and persist the patient snippets consumed by the NEPC classifier."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_NEPC.snippet_bundle import (  # noqa: E402
    SNIPPET_BUNDLE_FILENAME,
    write_snippet_bundle,
)
from shared.llm_helpers import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    PROSTATE_TEXT_CSV,
    build_patient_snippets,
    load_notes,
    load_selected_mrns,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean and trigger-scan prostate notes, rank patient snippets, and "
            "save the standalone artifact required by run_NEPC_classifier.py."
        )
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--notes-csv", type=Path, default=PROSTATE_TEXT_CSV)
    parser.add_argument("--note-bundle-path", type=Path, default=None)
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / SNIPPET_BUNDLE_FILENAME,
    )
    parser.add_argument("--max-notes-per-patient", type=int, default=75)
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
    if args.note_bundle_path is not None and args.note_bundle_path.exists():
        source_label, source_path = "bundle", args.note_bundle_path
    elif args.notes_csv.exists():
        source_label, source_path = "csv", args.notes_csv
    else:
        source_label, source_path = "raw_json", None
    print(
        "Note source: raw OncDRS JSONs"
        if source_path is None
        else f"Note source: {source_label} ({source_path})"
    )

    notes_df = load_notes(
        csv_path=args.notes_csv,
        bundle_path=args.note_bundle_path,
        raw_text_paths=args.raw_text_path,
        selected_mrns=selected_mrns,
    )
    all_mrns = {
        int(mrn) for mrn in notes_df["DFCI_MRN"].unique().drop_nulls().to_list()
    }
    print(f"Loaded notes: {len(notes_df)} rows for {len(all_mrns)} patients")

    patient_snippets = build_patient_snippets(
        notes_df,
        max_notes_per_patient=args.max_notes_per_patient,
    )
    write_snippet_bundle(
        args.output_path,
        all_mrns=all_mrns,
        patient_snippets=patient_snippets,
        metadata={
            "note_source": source_label,
            "note_source_path": None if source_path is None else str(source_path),
            "max_notes_per_patient": args.max_notes_per_patient,
        },
    )

    print(f"Wrote patient snippet bundle: {args.output_path}")
    print(f"Patients with triggered snippets: {len(patient_snippets)}")
    print(f"Patients with no signal: {len(all_mrns - set(patient_snippets))}")
    print(f"Total saved snippets: {sum(map(len, patient_snippets.values()))}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
