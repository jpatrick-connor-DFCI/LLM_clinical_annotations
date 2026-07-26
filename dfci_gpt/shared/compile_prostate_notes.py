"""Standalone prostate note extractor.

Extracts every available clinical note for a prostate MRN list from the raw
OncDRS clinical-text sources and writes a single `prostate_text_data.csv`, which
is the default note source for all downstream LLM pipelines (NEPC classifier,
Gleason timeline, AVPC/NEPC criteria timeline).

The default cohort source is the COMPASS prostate survival cohort file. The
`DFCI_MRN` column from that file defines which patients are included when no
explicit MRN list is supplied.
Raw note extraction streams the source JSONs via `ijson` when available.

Examples
--------
# Extract notes for an explicit MRN list
python shared/compile_prostate_notes.py --mrn-file prostate_mrns.txt

# Run with defaults: read MRNs from the COMPASS prostate survival cohort, then extract raw OncDRS notes
python shared/compile_prostate_notes.py
"""

import argparse
import sys
from pathlib import Path

import polars as pl

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

DEFAULT_PROSTATE_MRN_SOURCE = Path(
    "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/prostate_arpi_survival_cohort.csv"
)


def derive_prostate_mrns(cohort_source):
    cohort_source = Path(cohort_source)
    if not cohort_source.exists():
        raise FileNotFoundError(f"Cohort source not found: {cohort_source}")
    cohort = pl.scan_csv(cohort_source).select("DFCI_MRN").collect()
    return parse_mrn_values(cohort["DFCI_MRN"].to_list())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract raw OncDRS notes into prostate_text_data.csv. By default, "
        "the cohort MRNs are read from the COMPASS prostate survival cohort file, "
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
        help="Also union in the default cohort-source MRNs when --mrns/--mrn-file "
        "is provided. This happens automatically when no explicit MRNs are supplied.",
    )
    parser.add_argument(
        "--cohort-source",
        type=Path,
        default=DEFAULT_PROSTATE_MRN_SOURCE,
        help="CSV source whose DFCI_MRN column defines the default prostate cohort.",
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
        selected_mrns |= derive_prostate_mrns(args.cohort_source)
    if not selected_mrns:
        raise ValueError(
            "No MRNs selected. Provide --mrns/--mrn-file, or let the default "
            "cohort-source MRN inference run from --cohort-source."
        )

    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)
    note_df = load_raw_text_notes(raw_text_paths, selected_mrns)
    standardized = write_notes_csv(args.output_path, note_df)

    print(f"Wrote prostate notes CSV: {args.output_path}")
    print(f"Cohort MRNs requested: {len(selected_mrns)}")
    print(f"Patients with notes: {standardized['DFCI_MRN'].n_unique()}")
    print(f"Notes written: {len(standardized)}")
    print(f"Cohort source used: {args.cohort_source}")
    print(f"Raw text directories searched: {', '.join(str(p) for p in raw_text_paths)}")


if __name__ == "__main__":
    main()
