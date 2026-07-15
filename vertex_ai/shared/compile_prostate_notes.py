"""Standalone prostate note extractor.

Extracts every available clinical note for a prostate MRN list from the raw
OncDRS clinical-text sources and writes a single `prostate_text_data.csv`, which
is the default note source for all downstream LLM pipelines (NEPC classifier,
Gleason timeline, AVPC/NEPC criteria timeline).

The default cohort definition mirrors COMPASS's ICD-based rule: any patient
with ICD-10 C61, excluding patients with a competing non-prostate primary ICD.
Raw note extraction streams the source JSONs via `ijson` when available.

Examples
--------
# Extract notes for an explicit MRN list
python shared/compile_prostate_notes.py --mrn-file prostate_mrns.txt

# Run with defaults: derive the prostate cohort from raw OncDRS ICDs, then extract raw OncDRS notes
python shared/compile_prostate_notes.py
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

# Matches the raw OncDRS path used by
# PROFILE-testing/COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py.
DEFAULT_ICD_SOURCE = Path("/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/EHR_DIAGNOSIS.csv")


def mark_non_prostate_primary_icd(icds):
    """Mirror the COMPASS ICD exclusion rule for competing non-prostate primaries."""
    icds = icds.copy()
    codes = icds["DIAGNOSIS_ICD10_CD"].astype(str).str.upper().str.strip()

    letter = codes.str.extract(r"^([A-Z])", expand=False)
    number = pd.to_numeric(
        codes.str.extract(r"^[A-Z](\d{2,3})", expand=False), errors="coerce"
    )

    is_c00_c76 = (letter == "C") & (number >= 0) & (number <= 76)
    is_c81_c96 = (letter == "C") & (number >= 81) & (number <= 96)
    is_c97 = codes.str.startswith("C97")
    is_c7a = codes.str.startswith("C7A")
    is_c801 = codes.str.startswith("C801") | codes.str.startswith("C80.1")

    is_primary = is_c00_c76 | is_c81_c96 | is_c97 | is_c7a | is_c801
    is_prostate = codes.str.startswith("C61")
    is_secondary = ((letter == "C") & (number >= 77) & (number <= 79)) | codes.str.startswith("C7B")
    is_nmsc = codes.str.startswith("C44")
    is_nos = codes.str.startswith("C80.9") | codes.str.startswith("C809")

    icds["NON_PROSTATE_PRIMARY_ICD10"] = (
        is_primary & ~is_prostate & ~is_secondary & ~is_nmsc & ~is_nos
    )
    return icds


def load_and_explode_icd(icd_source):
    icd_source = Path(icd_source)
    if not icd_source.exists():
        raise FileNotFoundError(f"ICD source not found: {icd_source}")
    icds = pd.read_csv(icd_source)
    if "DIAGNOSIS_ICD10_LIST" in icds.columns and "DIAGNOSIS_ICD10_CD" not in icds.columns:
        icds["DIAGNOSIS_ICD10_CD"] = icds["DIAGNOSIS_ICD10_LIST"].astype(str).str.split(",")
        icds = icds.explode("DIAGNOSIS_ICD10_CD")
        icds["DIAGNOSIS_ICD10_CD"] = (
            icds["DIAGNOSIS_ICD10_CD"].astype(str).str.strip().str.upper()
        )
        icds = icds.loc[icds["DIAGNOSIS_ICD10_CD"] != ""]
    return icds


def derive_prostate_mrns(icd_source):
    icds = load_and_explode_icd(icd_source)
    codes = icds["DIAGNOSIS_ICD10_CD"].astype(str).str.upper().str.strip()
    c61_mrns = set(
        pd.to_numeric(icds.loc[codes.str.startswith("C61"), "DFCI_MRN"], errors="coerce")
        .dropna()
        .astype(int)
    )

    marked = mark_non_prostate_primary_icd(icds)
    non_prostate_primary_mrns = set(
        pd.to_numeric(
            marked.loc[marked["NON_PROSTATE_PRIMARY_ICD10"], "DFCI_MRN"], errors="coerce"
        )
        .dropna()
        .astype(int)
    )
    return c61_mrns - (c61_mrns & non_prostate_primary_mrns)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract raw OncDRS notes into prostate_text_data.csv. By default, "
        "the cohort is inferred from raw OncDRS ICDs using the COMPASS ICD-C61 rule, "
        "and note JSONs are streamed with ijson when available."
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
        help="Also union in the ICD-C61 cohort definition when --mrns/--mrn-file "
        "is provided. This happens automatically when no explicit MRNs are supplied.",
    )
    parser.add_argument(
        "--icd-source",
        type=Path,
        default=DEFAULT_ICD_SOURCE,
        help="ICD source used for the COMPASS-style ICD-C61 cohort definition "
        "(defaults to raw OncDRS EHR_DIAGNOSIS.csv).",
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
        selected_mrns |= derive_prostate_mrns(args.icd_source)
    if not selected_mrns:
        raise ValueError(
            "No MRNs selected. Provide --mrns/--mrn-file, or let the default "
            "raw OncDRS ICD cohort inference run from --icd-source."
        )

    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)
    note_df = load_raw_text_notes(raw_text_paths, selected_mrns)
    standardized = write_notes_csv(args.output_path, note_df)

    print(f"Wrote prostate notes CSV: {args.output_path}")
    print(f"Cohort MRNs requested: {len(selected_mrns)}")
    print(f"Patients with notes: {standardized['DFCI_MRN'].nunique()}")
    print(f"Notes written: {len(standardized)}")
    print(f"ICD source used: {args.icd_source}")
    print(f"Raw text directories searched: {', '.join(str(p) for p in raw_text_paths)}")


if __name__ == "__main__":
    main()
