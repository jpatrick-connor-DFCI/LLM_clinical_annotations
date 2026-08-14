"""Stage 1 — Collect notes mentioning NEPC diagnosis language (strict variant).

Scans notes for every prostate patient, groups matches into per-patient,
payload-sized chunks, and writes them as evidence for the LLM step.

This is the strict-precision sibling of collect_nepc_notes.py. It retrieves a
much narrower candidate pool (see TRIGGER_REGEX below) because the downstream
task answers one question -- does the record state an NEPC diagnosis, and when
-- rather than assembling composite Aparicio criteria from atomic facts.

Outputs (under <output-dir>):
  nepc_dx_evidence.parquet        one row per snippet, grouped by patient/chunk
  nepc_dx_evidence.meta.parquet   scan_config hash + resolved params the evidence
                                  was built under (used by build_nepc_dx_labels.py
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
    NEPC_DX_EVIDENCE_SCHEMA_VERSION,
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

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_nepc_diagnosis"
_PROFILE = SNIPPET_PROFILES["longitudinal"]

# Far narrower than preprocessing.triggers.TRIGGER_REGEX["nepc"]. Dropped versus
# the longitudinal collector:
#   - the entire "avpc" and "avpc_atomic" families: Aparicio composite criteria
#     are out of scope here, and avpc_atomic matches nearly every prostate note
#     via psa/calcium/gleason;
#   - bare transform(ation|ed|ing), which matches "transformation of care" and
#     "transformed lymphoma";
#   - standalone "nse";
#   - dedifferentiat* and "lineage plasticity", which are biology-discussion
#     language rather than an asserted diagnosis.
#
# IHC marker names are KEPT as retrieval triggers only -- a pathology report
# whose diagnosis line reads "small cell carcinoma" almost always also stains,
# and the stain line carries useful surrounding context. They can never
# establish a positive on their own: tasks/nepc_diagnosis/veto.py requires a
# diagnostic assertion in the quote, so an IHC-only note is retrieved and then
# rejected.
#
# A single trigger label is used deliberately: evidence_scan_config_key hashes
# each label together with its pattern, so this evidence can never be confused
# with the longitudinal collector's.
TRIGGER_REGEX = {
    "nepc_dx": (
        r"(?:"
        r"\b(?:nepc|t[\s-]?nepc|scpc|scnc)\b|"
        r"\bneuro[\s-]?endocrine\b|"
        r"\bsmall[\s-]?cell\b|"
        r"\boat[\s-]?cell\b|"
        r"\bhistolog(?:ic|ical)(?:ally)?\s+transform(?:ation|ed|ing)\b|"
        r"\btransform(?:ation|ed|ing)\s+(?:in)?to\s+(?:a\s+|an\s+)?"
        r"(?:small[\s-]?cell|neuro[\s-]?endocrine|nepc)\b|"
        r"\btransdifferentiat(?:e|ed|ion|ing)\b|"
        r"\btreatment[\s-]?emergent\s+neuro[\s-]?endocrine\b|"
        r"\b(?:synaptophysin|chromogranin(?:\s+a)?|cd56|insm1)\b|"
        r"\bneuron[\s-]specific\s+enolase\b"
        r")"
    ),
}

EVIDENCE_COLUMNS = ["DFCI_MRN", "chunk_index", "note_date", "note_type", "snippet"]

# Stage-2 artifacts discarded by --overwrite, since regenerated evidence
# invalidates the chunk_index assignments they resume against.
STAGE2_ARTIFACTS = (
    "nepc_dx_candidates_raw.parquet",
    "nepc_dx_labels.parquet",
    "nepc_dx_processed_chunks.parquet",
    "nepc_dx_processed_patients.parquet",
    "nepc_dx_rejected_findings.parquet",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect notes mentioning NEPC diagnosis language (strict variant)."
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
        default=1200,
        help="Chars of context kept on each side of an NEPC match. Narrower than the "
        "AVPC collector's window: this task only needs the sentence carrying the "
        "diagnostic assertion plus the negation span before it.",
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
    progress = tqdm(total=4, desc="Compile NEPC diagnosis evidence", unit="step", dynamic_ncols=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "nepc_dx_evidence.parquet"
    meta_path = args.output_dir / "nepc_dx_evidence.meta.parquet"
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
            "NEPC diagnosis extraction is prostate-specific. Direct "
            "PROFILE_DATA parquet runs require --mrns or --mrn-file to define "
            "the prostate cohort."
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
                "NEPC diagnosis scan settings differ from the existing evidence "
                f"({existing_meta.get('scan_config')} != {scan_config}). "
                "Re-run with --overwrite instead of mixing incompatible evidence "
                "and chunk state."
            )
        if (
            existing_meta.get("evidence_schema_version")
            != NEPC_DX_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Existing NEPC diagnosis evidence predates the current "
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
        f"Patients mentioning NEPC diagnosis language: {len(patient_chunks)} "
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
        evidence_schema_version=NEPC_DX_EVIDENCE_SCHEMA_VERSION,
        cohort_mrn_count=len(selected_mrns) if selected_mrns is not None else None,
        # Preserve the actual denominator, not only its size, so stage 2 can
        # materialize explicit auto-negative rows for cohort patients who never
        # produced a trigger-bearing evidence chunk.
        cohort_mrns=sorted(selected_mrns) if selected_mrns is not None else None,
    )
    progress.update(1)
    progress.close()
    print(f"Wrote NEPC diagnosis evidence ({evidence.height} rows): {evidence_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
