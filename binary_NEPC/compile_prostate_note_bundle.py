import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.llm_helpers import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    NOTE_BUNDLE_FILENAME,
    load_raw_text_notes,
    load_selected_mrns,
    resolve_raw_text_paths,
    write_note_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile all raw notes for a prostate MRN list into a gzip JSON bundle for binary_NEPC."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / NOTE_BUNDLE_FILENAME,
        help="Destination gzip JSON note bundle to write.",
    )
    parser.add_argument(
        "--raw-text-path",
        type=Path,
        action="append",
        default=None,
        help="Raw OncDRS note directory. Repeat to search multiple directories.",
    )
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        required=True,
        help="Text/CSV/TSV file containing the prostate DFCI_MRN values to compile.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    if selected_mrns is None:
        raise ValueError("A non-empty MRN selection is required.")

    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)
    note_df = load_raw_text_notes(raw_text_paths, selected_mrns)
    write_note_bundle(
        args.output_path,
        note_df,
        raw_text_paths=raw_text_paths,
        selected_mrns=selected_mrns,
    )

    print(f"Wrote compiled note bundle: {args.output_path}")
    print(f"Patients in bundle: {note_df['DFCI_MRN'].nunique()}")
    print(f"Notes in bundle: {len(note_df)}")
    print(f"Requested MRNs: {len(selected_mrns)}")
    print(f"Raw text directories searched: {', '.join(str(path) for path in raw_text_paths)}")


if __name__ == "__main__":
    main()
