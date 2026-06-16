"""Standalone prostate note extractor.

Extracts every available clinical note for a prostate MRN list from the raw
OncDRS clinical-text sources and writes a single `prostate_text_data.csv`, which
is the default note source for all downstream LLM pipelines (NEPC classifier,
Gleason timeline, AVPC/NEPC criteria timeline).

This replaces the deprecated batched-VTE note compilation that previously
produced prostate_text_data.csv.

Examples
--------
# Extract notes for an explicit MRN list
python shared/compile_prostate_notes.py --mrn-file prostate_mrns.txt

# Derive the prostate cohort from the inferred-cancer table, then extract
python shared/compile_prostate_notes.py --derive-prostate-mrns
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.llm_helpers import (  # noqa: E402
    PROSTATE_TEXT_CSV,
    load_raw_text_notes,
    load_selected_mrns,
    parse_mrn_values,
    resolve_raw_text_paths,
    write_notes_csv,
)

# Inferred-cancer cohort table used to derive the prostate MRN set when no
# explicit MRN list is supplied (mirrors data_preprocessing/compile_prostate_data.py).
DEFAULT_COHORT_CSV = Path(
    "/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/"
    "first_treatments_dfci_w_inferred_cancers.csv"
)
PROSTATE_CANCER_GROUP = "PROSTATE"


def derive_prostate_mrns(cohort_csv):
    cohort_csv = Path(cohort_csv)
    if not cohort_csv.exists():
        raise FileNotFoundError(f"Cohort table not found: {cohort_csv}")
    cohort = pd.read_csv(
        cohort_csv, usecols=["DFCI_MRN", "med_genomics_merged_cancer_group"]
    )
    prostate = cohort.loc[
        cohort["med_genomics_merged_cancer_group"] == PROSTATE_CANCER_GROUP, "DFCI_MRN"
    ]
    return parse_mrn_values(prostate)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all raw OncDRS notes for a prostate MRN list into prostate_text_data.csv."
    )
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Text/CSV/TSV file with the prostate DFCI_MRN values to compile.",
    )
    parser.add_argument(
        "--derive-prostate-mrns",
        action="store_true",
        help="Derive the prostate MRN set from the inferred-cancer cohort table "
        "(used when no --mrns/--mrn-file is given, or to union with them).",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=DEFAULT_COHORT_CSV,
        help="Inferred-cancer cohort table for --derive-prostate-mrns.",
    )
    parser.add_argument(
        "--raw-text-path",
        type=Path,
        action="append",
        default=None,
        help="Raw OncDRS note directory. Repeat to search multiple directories.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROSTATE_TEXT_CSV,
        help="Destination CSV (default: the shared prostate_text_data.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file) or set()
    selected_mrns = set(selected_mrns)
    if args.derive_prostate_mrns or not selected_mrns:
        selected_mrns |= derive_prostate_mrns(args.cohort_csv)
    if not selected_mrns:
        raise ValueError(
            "No MRNs selected. Provide --mrns/--mrn-file or use --derive-prostate-mrns."
        )

    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)
    note_df = load_raw_text_notes(raw_text_paths, selected_mrns)
    standardized = write_notes_csv(args.output_path, note_df)

    print(f"Wrote prostate notes CSV: {args.output_path}")
    print(f"Requested MRNs: {len(selected_mrns)}")
    print(f"Patients with notes: {standardized['DFCI_MRN'].nunique()}")
    print(f"Notes written: {len(standardized)}")
    print(f"Raw text directories searched: {', '.join(str(p) for p in raw_text_paths)}")


if __name__ == "__main__":
    main()
