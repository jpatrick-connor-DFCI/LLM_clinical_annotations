"""Standalone prostate note extractor.

Selects clinical notes for a prostate MRN list from the merged PROFILE_DATA
parquets and writes a `prostate_text_data.parquet` artifact.

The default cohort source is the COMPASS prostate survival cohort file. The
`DFCI_MRN` column from that file defines which patients are included when no
explicit MRN list is supplied.

Examples
--------
# Extract notes for an explicit MRN list
python preprocessing/cli/compile_prostate_notes.py --mrn-file prostate_mrns.csv

# Run with defaults: read cohort MRNs, then select their PROFILE_DATA notes
python preprocessing/cli/compile_prostate_notes.py
"""

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import (  # noqa: E402
    DEFAULT_ICD_PROSTATE_MRN_CSV,
    DEFAULT_PROFILE_NOTE_PATHS,
    PROSTATE_TEXT_PARQUET,
)
from preprocessing.notes import (  # noqa: E402
    load_profile_notes,
    load_selected_mrns,
    parse_mrn_values,
    write_notes_parquet,
)

DEFAULT_PROSTATE_MRN_SOURCE = DEFAULT_ICD_PROSTATE_MRN_CSV


def derive_prostate_mrns(cohort_source):
    cohort_source = Path(cohort_source)
    if not cohort_source.exists():
        raise FileNotFoundError(f"Cohort source not found: {cohort_source}")
    if cohort_source.suffix.lower() == ".csv":
        cohort = pl.scan_csv(cohort_source).select("DFCI_MRN").collect()
    elif cohort_source.suffix.lower() == ".parquet":
        cohort = pl.scan_parquet(cohort_source).select("DFCI_MRN").collect()
    else:
        raise ValueError(f"Cohort source must be CSV or Parquet: {cohort_source}")
    return parse_mrn_values(cohort["DFCI_MRN"].to_list())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select PROFILE_DATA notes into prostate_text_data.parquet. "
        "By default, cohort MRNs come from the COMPASS prostate survival cohort."
    )
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="CSV cohort file with the prostate DFCI_MRN values to compile.",
    )
    parser.add_argument(
        "--derive-prostate-mrns",
        action="store_true",
        help="Also union in the default cohort-source MRNs when --mrns/--mrn-file "
        "is provided. This happens automatically when no explicit MRNs are supplied.",
    )
    parser.add_argument(
        "--cohort-source",
        type=Path,
        default=DEFAULT_PROSTATE_MRN_SOURCE,
        help="CSV source whose DFCI_MRN column defines the default prostate cohort; "
             "defaults to COMPASS_PROFILE_DATA/mrn_lists/icd_prostate_mrn_flags.csv.",
    )
    parser.add_argument(
        "--notes-parquet",
        type=Path,
        action="append",
        default=None,
        help="PROFILE_DATA clinical-note parquet. Repeat for multiple files; defaults "
        "to pathology, imaging, and progress notes.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROSTATE_TEXT_PARQUET,
        help="Destination Parquet (default: the shared prostate_text_data.parquet).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file) or set()
    selected_mrns = set(selected_mrns)
    if args.derive_prostate_mrns or not selected_mrns:
        selected_mrns |= derive_prostate_mrns(args.cohort_source)
    if not selected_mrns:
        raise ValueError(
            "No MRNs selected. Provide --mrns/--mrn-file, or let the default "
            "cohort-source MRN inference run from --cohort-source."
        )

    parquet_paths = args.notes_parquet or DEFAULT_PROFILE_NOTE_PATHS
    note_df = load_profile_notes(parquet_paths, selected_mrns)
    standardized = write_notes_parquet(args.output_path, note_df)

    print(f"Wrote prostate notes Parquet: {args.output_path}")
    print(f"Cohort MRNs requested: {len(selected_mrns)}")
    print(f"Patients with notes: {standardized['DFCI_MRN'].n_unique()}")
    print(f"Notes written: {len(standardized)}")
    print(f"Cohort source used: {args.cohort_source}")
    print(f"PROFILE_DATA parquets read: {', '.join(str(p) for p in parquet_paths)}")


if __name__ == "__main__":
    main()
