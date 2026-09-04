"""Stage 1 — Collect notes mentioning metastatic disease language.

Scans notes for every prostate patient, groups matches into per-patient,
payload-sized chunks, and writes them as evidence for the LLM step.

The downstream task answers one question -- does the record assert distant
metastatic prostate cancer, and when was it first documented -- so retrieval
here is deliberately BROAD and precision is deferred to
tasks/met_diagnosis/veto.py. That inversion is the opposite of what the trigger
set alone suggests: `metasta\\w*` fires on every "no evidence of metastatic
disease" in every surveillance scan, and `bone scan` fires on the negative ones
too. Those notes must be retrieved anyway, because the negation is exactly what
the gate needs to see in order to reject them; a narrower trigger set would
silently miss the positives sitting next to them in the same report.

Outputs (under <output-dir>):
  met_dx_evidence.parquet        one row per snippet, grouped by patient/chunk
  met_dx_evidence.meta.parquet   scan_config hash + resolved params the evidence
                                 was built under (used by build_met_dx_labels.py
                                 to validate that chunk-index resume is safe)

Re-running without --overwrite: if the evidence file's recorded scan_config
matches the resolved settings for this invocation, the scan is skipped entirely
(existing evidence is reused as-is). If it differs, this raises -- regenerating
evidence with different scan parameters changes chunk_index assignment, which
would silently corrupt stage-2's chunk-level resume; pass --overwrite to
intentionally rescan and discard the old evidence + chunk state.
"""

import argparse
import sys
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_ADT_MRN_CSV,
    DEFAULT_PROFILE_NOTE_PATHS,
    MET_DX_EVIDENCE_SCHEMA_VERSION,
    SNIPPET_PROFILES,
)
from preprocessing.longitudinal import (  # noqa: E402
    evidence_scan_config_key,
    file_sha256,
    filter_note_types,
    group_patient_snippets,
    read_scan_config_meta,
    write_scan_config_meta,
)
from preprocessing.notes import load_notes, load_selected_mrns  # noqa: E402
from preprocessing.parquet_io import write_parquet_atomic  # noqa: E402
from preprocessing.triggers import combined_text_pattern  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_met_diagnosis"
_PROFILE = SNIPPET_PROFILES["longitudinal"]

# Broader than preprocessing.triggers.TRIGGER_REGEX["avpc"], which matches only
# SITE-QUALIFIED met phrases ("liver metastases", "lytic bone lesion") because it
# is assembling Aparicio criteria rather than asking whether the patient is
# metastatic at all. Bare metasta*, mets, M1*, and stage IV are added here --
# they are the most common way the fact is actually written, and none of them
# match any existing trigger family.
#
# Lesion-descriptor triggers (osseous/sclerotic/lytic lesions, bone scan) are
# retrieval-only: they carry no assertion of malignancy by themselves, and
# veto.py requires an assertion anchor, so a degenerative-changes report is
# retrieved and then rejected. They earn their place because the first
# documentation of bone metastasis is frequently a bone scan impression that
# never uses the word "metastatic" in the sentence a narrow window would keep.
#
# A single trigger label is used deliberately: evidence_scan_config_key hashes
# each label together with its pattern, so this evidence can never be confused
# with another collector's.
TRIGGER_REGEX = {
    "met_dx": (
        r"(?:"
        r"\bmetasta\w*\b|"
        r"\bmets\b|"
        r"(?:\b|(?<=\d))m1[abc]?\b|"
        r"\bstage\s+(?:iv|4)\b|"
        r"\bdistant\s+(?:disease|spread|sites?|metasta\w*)\b|"
        r"\bwidespread\s+(?:disease|osseous|skeletal)\b|"
        r"\bdisseminated\s+disease\b|"
        r"\b(?:osseous|skeletal|bone)\s+(?:lesions?|involvement|disease|uptake)\b|"
        r"\bsclerotic\s+(?:bone\s+)?lesions?\b|"
        r"\blytic\s+(?:bone\s+)?lesions?\b|"
        r"\bbone\s+scan\b|"
        r"\bcarcinomatosis\b|"
        r"\bsecondary\s+(?:malignant\s+)?neoplasm\b"
        r")"
    ),
}

EVIDENCE_COLUMNS = ["DFCI_MRN", "chunk_index", "note_date", "note_type", "snippet"]

# Stage-2 artifacts discarded by --overwrite, since regenerated evidence
# invalidates the chunk_index assignments they resume against.
STAGE2_ARTIFACTS = (
    "met_dx_candidates_raw.parquet",
    "met_dx_labels.parquet",
    "met_dx_processed_chunks.parquet",
    "met_dx_processed_patients.parquet",
    "met_dx_rejected_findings.parquet",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect notes mentioning metastatic disease language."
    )
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=DEFAULT_ADT_MRN_CSV,
        help="CSV cohort file containing DFCI_MRN values; defaults to the "
             "COMPASS ADT cohort.",
    )
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--notes-parquet", type=Path, action="append", default=None,
                        help="PROFILE_DATA clinical-note parquet. Repeat for multiple files; "
                             "defaults to pathology, imaging, and progress notes.")
    parser.add_argument("--note-bundle-path", type=Path, default=None,
                        help="Standardized Parquet note bundle override.")
    parser.add_argument(
        "--note-types",
        nargs="+",
        default=None,
        help="Restrict to these NOTE_TYPE values (e.g. Pathology Imaging). Default: all notes.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=1500,
        help="Chars of context kept on each side of a metastasis match. Wider than the "
        "NEPC diagnosis collector's window: radiology negation spans are long "
        "(\"no new osseous lesions ... unchanged from prior ... no evidence of "
        "metastatic disease\"), and the site attribution that separates a distant "
        "met from regional nodal disease often sits a sentence away from the term.",
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
    args = parser.parse_args()
    if args.context_chars < 0:
        parser.error("--context-chars must be >= 0")
    if args.payload_max_chars < _PROFILE.max_chars:
        parser.error(
            f"--payload-max-chars must be >= the per-snippet cap ({_PROFILE.max_chars})"
        )
    if args.scan_workers is not None and args.scan_workers < 1:
        parser.error("--scan-workers must be >= 1")
    return args


def run(args):
    progress = tqdm(total=4, desc="Compile metastasis evidence", unit="step", dynamic_ncols=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "met_dx_evidence.parquet"
    meta_path = args.output_dir / "met_dx_evidence.meta.parquet"
    snippet_max_chars = _PROFILE.max_chars

    if args.context_chars < 0:
        raise ValueError("context_chars must be >= 0")
    if args.payload_max_chars < snippet_max_chars:
        raise ValueError(
            f"payload_max_chars must be >= snippet_max_chars ({snippet_max_chars})"
        )
    if args.scan_workers is not None and args.scan_workers < 1:
        raise ValueError("scan_workers must be >= 1")

    if args.overwrite:
        for path in (
            evidence_path,
            meta_path,
            *(args.output_dir / name for name in STAGE2_ARTIFACTS),
        ):
            path.unlink(missing_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    direct_parquet = args.note_bundle_path is None
    if direct_parquet and selected_mrns is None:
        raise ValueError(
            "Metastasis extraction is prostate-specific. Direct PROFILE_DATA "
            "parquet runs require --mrns or --mrn-file to define the prostate "
            "cohort."
        )
    notes_df = load_notes(
        parquet_paths=(args.notes_parquet or DEFAULT_PROFILE_NOTE_PATHS)
        if args.note_bundle_path is None else None,
        bundle_path=args.note_bundle_path,
        selected_mrns=selected_mrns,
        text_pattern=combined_text_pattern(TRIGGER_REGEX),
        note_types=args.note_types,
    )
    print(
        f"Loaded notes: {len(notes_df)} rows for "
        f"{notes_df['DFCI_MRN'].n_unique()} patients"
    )
    progress.update(1)

    if args.note_types and args.note_bundle_path is not None:
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
    progress.update(1)

    if evidence_path.exists() and evidence_path.stat().st_size > 0:
        existing_meta = read_scan_config_meta(meta_path)
        if existing_meta is None:
            raise ValueError(
                f"Existing evidence has no scan-config sidecar: {meta_path}. "
                "Re-run with --overwrite to rebuild it safely."
            )
        if existing_meta.get("scan_config") != scan_config:
            raise ValueError(
                "Metastasis scan settings differ from the existing evidence "
                f"({existing_meta.get('scan_config')} != {scan_config}). "
                "Re-run with --overwrite instead of mixing incompatible evidence "
                "and chunk state."
            )
        if (
            existing_meta.get("evidence_schema_version")
            != MET_DX_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Existing metastasis evidence predates the current "
                "trigger/cohort contract. Re-run with --overwrite."
            )
        recorded_digest = existing_meta.get("evidence_sha256")
        actual_digest = file_sha256(evidence_path)
        if not recorded_digest or recorded_digest != actual_digest:
            raise ValueError(
                "Existing evidence content does not match its metadata sidecar. "
                "Re-run with --overwrite to rebuild evidence and downstream state."
            )
        # The trigger-bearing evidence can be reused when only no-trigger cohort
        # membership changes, but keep the denominator itself current so stage 2
        # can materialize those patients as auto-negatives.
        current_cohort = (
            sorted(selected_mrns) if selected_mrns is not None else None
        )
        if existing_meta.get("cohort_mrns") != current_cohort:
            refreshed_meta = dict(existing_meta)
            refreshed_meta.pop("scan_config", None)
            refreshed_meta.update(
                cohort_mrn_count=(
                    len(selected_mrns) if selected_mrns is not None else None
                ),
                cohort_mrns=current_cohort,
            )
            write_scan_config_meta(meta_path, scan_config, **refreshed_meta)
        print(f"Existing evidence matches current scan settings, reusing: {evidence_path}")
        progress.update(2)
        progress.close()
        return

    patient_chunks = group_patient_snippets(
        notes_df,
        TRIGGER_REGEX,
        context_chars=args.context_chars,
        payload_max_chars=args.payload_max_chars,
        max_workers=args.scan_workers,
    )
    progress.update(1)
    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients mentioning metastatic disease language: {len(patient_chunks)} "
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
    write_parquet_atomic(evidence, evidence_path)
    write_scan_config_meta(
        meta_path,
        scan_config,
        context_chars=args.context_chars,
        snippet_max_chars=snippet_max_chars,
        payload_max_chars=args.payload_max_chars,
        note_types=args.note_types,
        evidence_sha256=file_sha256(evidence_path),
        evidence_schema_version=MET_DX_EVIDENCE_SCHEMA_VERSION,
        cohort_mrn_count=len(selected_mrns) if selected_mrns is not None else None,
        # Preserve the actual denominator, not only its size, so stage 2 can
        # materialize explicit auto-negative rows for cohort patients who never
        # produced a trigger-bearing evidence chunk.
        cohort_mrns=sorted(selected_mrns) if selected_mrns is not None else None,
    )
    progress.update(1)
    progress.close()
    print(f"Wrote metastasis evidence ({evidence.height} rows): {evidence_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
