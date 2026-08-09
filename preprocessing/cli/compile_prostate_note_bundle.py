import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import (  # noqa: E402
    DEFAULT_ADT_MRN_CSV,
    DEFAULT_PROFILE_NOTE_PATHS,
    DEFAULT_OUTPUT_DIR,
    NOTE_BUNDLE_FILENAME,
)
from preprocessing.notes import (  # noqa: E402
    load_notes,
    load_selected_mrns,
    write_note_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile prostate notes for a prostate MRN list into a Parquet bundle "
        "for binary_NEPC. Defaults to reading the merged PROFILE_DATA note parquets."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / NOTE_BUNDLE_FILENAME,
        help="Destination Parquet note bundle to write.",
    )
    parser.add_argument(
        "--notes-parquet",
        type=Path,
        action="append",
        default=None,
        help="PROFILE_DATA clinical-note parquet. Repeat for multiple files; defaults "
        "to pathology, imaging, and progress notes.",
    )
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=DEFAULT_ADT_MRN_CSV,
        help="CSV cohort file containing the prostate DFCI_MRN values; defaults "
             "to the COMPASS ADT cohort.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    progress = tqdm(total=3, desc="Compile note bundle", unit="step", dynamic_ncols=True)
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    progress.update(1)

    parquet_paths = args.notes_parquet or DEFAULT_PROFILE_NOTE_PATHS
    if selected_mrns is None:
        raise ValueError(
            "Compiling a prostate note bundle directly from PROFILE_DATA requires "
            "--mrns or --mrn-file to define the prostate cohort."
        )
    note_df = load_notes(
        parquet_paths=parquet_paths,
        bundle_path=None,
        selected_mrns=selected_mrns,
    )
    progress.update(1)
    write_note_bundle(
        args.output_path,
        note_df,
        selected_mrns=selected_mrns,
    )
    progress.update(1)
    progress.close()

    print(f"Wrote compiled note bundle: {args.output_path}")
    print(f"Patients in bundle: {note_df['DFCI_MRN'].n_unique()}")
    print(f"Notes in bundle: {len(note_df)}")
    print(f"Requested MRNs: {len(selected_mrns)}")
    print(f"Note source: {', '.join(str(path) for path in parquet_paths)}")


if __name__ == "__main__":
    main()
