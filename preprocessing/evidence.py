"""Per-note evidence collection shared by the note-level extraction pipelines.

The NEPC and AVPC pipelines make one LLM call per trigger-bearing note, so
their evidence is one row per deduplicated note rather than per-patient packed
chunks. `collect_note_evidence` generalizes `collect_nepc_notes.run` for that
shape and keeps its reuse/raise/overwrite contract:

  - unchanged scan settings reuse the existing evidence and skip the scan;
  - changed settings raise unless --overwrite is passed;
  - a changed cohort with unchanged trigger-bearing notes only refreshes the
    sidecar's cohort fields, so no-trigger patients stay in the denominator.

Evidence columns: DFCI_MRN, note_uid, note_date, note_type, snippet.
"""

from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from preprocessing.config import DEFAULT_PROFILE_NOTE_PATHS
from preprocessing.longitudinal import (
    evidence_scan_config_key,
    file_sha256,
    filter_note_types,
    iter_note_snippets,
    read_scan_config_meta,
    write_scan_config_meta,
)
from preprocessing.notes import load_notes, load_selected_mrns
from preprocessing.parquet_io import write_parquet_atomic
from preprocessing.triggers import combined_text_pattern

NOTE_EVIDENCE_COLUMNS = ["DFCI_MRN", "note_uid", "note_date", "note_type", "snippet"]


def evidence_meta_path(evidence_path):
    return evidence_path.with_name(f"{evidence_path.stem}.meta.parquet")


def note_task_paths(output_dir, prefix):
    """Artifact layout shared by a per-note collector and its task runner."""
    output_dir = Path(output_dir)
    return {
        "evidence": output_dir / f"{prefix}_evidence.parquet",
        "evidence_meta": output_dir / f"{prefix}_evidence.meta.parquet",
        "findings": output_dir / f"{prefix}_findings.parquet",
        "rejected": output_dir / f"{prefix}_rejected_findings.parquet",
        "processed": output_dir / f"{prefix}_processed_notes.parquet",
        "labels": output_dir / f"{prefix}_labels.parquet",
        "labels_meta": output_dir / f"{prefix}_labels.meta.parquet",
        "run": output_dir / f"{prefix}_run.parquet",
    }


def downstream_paths(output_dir, prefix):
    """Runner outputs that become stale when the evidence is rebuilt."""
    paths = note_task_paths(output_dir, prefix)
    return [paths[k] for k in ("findings", "rejected", "processed", "labels", "labels_meta", "run")]


def _unique_note_uids(records):
    """Give every (mrn, note) a stable, unique id.

    RAW_NOTE_ID is the source RPT_ID, which is only unique within one PROFILE
    parquet, so the note type is prefixed. A residual collision gets a "#n"
    suffix assigned in a deterministic order.
    """
    records = sorted(
        records,
        key=lambda r: (
            int(r["DFCI_MRN"]),
            r["note_date"] or "9999-99-99",
            r["note_type"] or "",
            str(r["note_uid"]),
            r["snippet"],
        ),
    )
    seen = {}
    for rec in records:
        base = f"{rec['note_type']}:{rec['note_uid']}"
        key = (int(rec["DFCI_MRN"]), base)
        count = seen.get(key, 0)
        seen[key] = count + 1
        rec["note_uid"] = base if count == 0 else f"{base}#{count}"
    return records


def summarize_evidence(evidence, label):
    """Print notes / characters per note type so LLM cost can be sized first."""
    if evidence.is_empty():
        print(f"{label}: no trigger-bearing notes")
        return
    summary = (
        evidence.with_columns(pl.col("snippet").str.len_chars().alias("_chars"))
        .group_by("note_type")
        .agg(
            pl.len().alias("notes"),
            pl.col("DFCI_MRN").n_unique().alias("patients"),
            pl.col("_chars").sum().alias("total_chars"),
            pl.col("_chars").mean().round(0).alias("mean_chars"),
            pl.col("_chars").max().alias("max_chars"),
        )
        .sort("note_type")
    )
    total_chars = int(summary["total_chars"].sum())
    print(
        f"{label}: {evidence.height} notes (one LLM call each) for "
        f"{evidence['DFCI_MRN'].n_unique()} patients, {total_chars:,} snippet chars "
        f"(~{total_chars // 4:,} input tokens before the system prompt)"
    )
    for row in summary.iter_rows(named=True):
        print(
            f"  {row['note_type']:<10} notes={row['notes']:>8,} "
            f"patients={row['patients']:>7,} chars={row['total_chars']:>13,} "
            f"mean={int(row['mean_chars']):>7,} max={row['max_chars']:>7,}"
        )


def collect_note_evidence(
    args,
    *,
    trigger_regex,
    evidence_path,
    schema_version,
    snippet_max_chars,
    downstream_paths=(),
    context_overrides=None,
    label="Evidence",
    cohort_required_message=None,
):
    """Scan notes into one-row-per-note evidence plus its metadata sidecar.

    `args` carries the collector CLI fields: mrns, mrn_file, notes_parquet,
    note_bundle_path, note_types, context_chars, scan_workers, overwrite.
    `downstream_paths` are the task runner's outputs, removed on --overwrite
    because they were produced from the evidence being discarded.
    Returns the evidence path.
    """
    if args.context_chars < 0:
        raise ValueError("context_chars must be >= 0")
    if args.scan_workers is not None and args.scan_workers < 1:
        raise ValueError("scan_workers must be >= 1")

    progress = tqdm(total=4, desc=f"Compile {label}", unit="step", dynamic_ncols=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = evidence_meta_path(evidence_path)

    if args.overwrite:
        for path in (evidence_path, meta_path, *downstream_paths):
            path.unlink(missing_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    if args.note_bundle_path is None and selected_mrns is None:
        raise ValueError(
            cohort_required_message
            or (
                f"{label} extraction is prostate-specific. Direct PROFILE_DATA "
                "parquet runs require --mrns or --mrn-file to define the cohort."
            )
        )
    notes_df = load_notes(
        parquet_paths=(args.notes_parquet or DEFAULT_PROFILE_NOTE_PATHS)
        if args.note_bundle_path is None else None,
        bundle_path=args.note_bundle_path,
        selected_mrns=selected_mrns,
        text_pattern=combined_text_pattern(trigger_regex),
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

    # No chunk packing here, so the per-note cap is also the per-call budget.
    scan_config = evidence_scan_config_key(
        notes_df,
        trigger_regex,
        context_chars=args.context_chars,
        snippet_max_chars=snippet_max_chars,
        payload_max_chars=snippet_max_chars,
        note_types=args.note_types,
        context_overrides=context_overrides,
        preserve_trial_context=True,
    )
    progress.update(1)

    current_cohort = sorted(selected_mrns) if selected_mrns is not None else None
    if evidence_path.exists() and evidence_path.stat().st_size > 0:
        existing_meta = read_scan_config_meta(meta_path)
        if existing_meta is None:
            raise ValueError(
                f"Existing evidence has no scan-config sidecar: {meta_path}. "
                "Re-run with --overwrite to rebuild it safely."
            )
        if existing_meta.get("scan_config") != scan_config:
            raise ValueError(
                f"{label} scan settings differ from the existing evidence "
                f"({existing_meta.get('scan_config')} != {scan_config}). "
                "Re-run with --overwrite instead of mixing incompatible evidence "
                "and note state."
            )
        if existing_meta.get("evidence_schema_version") != schema_version:
            raise ValueError(
                f"Existing {label} evidence predates the current trigger/cohort "
                "contract. Re-run with --overwrite."
            )
        recorded_digest = existing_meta.get("evidence_sha256")
        if not recorded_digest or recorded_digest != file_sha256(evidence_path):
            raise ValueError(
                "Existing evidence content does not match its metadata sidecar. "
                "Re-run with --overwrite to rebuild evidence and downstream state."
            )
        if existing_meta.get("cohort_mrns") != current_cohort:
            refreshed_meta = dict(existing_meta)
            refreshed_meta.pop("scan_config", None)
            refreshed_meta.update(
                cohort_mrn_count=len(current_cohort) if current_cohort is not None else None,
                cohort_mrns=current_cohort,
            )
            write_scan_config_meta(meta_path, scan_config, **refreshed_meta)
        print(f"Existing evidence matches current scan settings, reusing: {evidence_path}")
        summarize_evidence(pl.read_parquet(evidence_path), label)
        progress.update(2)
        progress.close()
        return evidence_path

    records = list(
        iter_note_snippets(
            notes_df,
            trigger_regex,
            context_chars=args.context_chars,
            snippet_max_chars=snippet_max_chars,
            max_workers=args.scan_workers,
            context_overrides=context_overrides,
            preserve_trial_context=True,
        )
    )
    progress.update(1)
    records = _unique_note_uids(records)

    if records:
        evidence = pl.DataFrame(
            {
                "DFCI_MRN": [int(r["DFCI_MRN"]) for r in records],
                "note_uid": [r["note_uid"] for r in records],
                "note_date": [r["note_date"] for r in records],
                "note_type": [r["note_type"] for r in records],
                "snippet": [r["snippet"] for r in records],
            },
            schema={
                "DFCI_MRN": pl.Int64,
                "note_uid": pl.Utf8,
                "note_date": pl.Utf8,
                "note_type": pl.Utf8,
                "snippet": pl.Utf8,
            },
        )
    else:
        evidence = pl.DataFrame(
            schema={"DFCI_MRN": pl.Int64, **{c: pl.Utf8 for c in NOTE_EVIDENCE_COLUMNS[1:]}}
        )
    write_parquet_atomic(evidence, evidence_path)
    write_scan_config_meta(
        meta_path,
        scan_config,
        context_chars=args.context_chars,
        snippet_max_chars=snippet_max_chars,
        context_overrides=repr(sorted(context_overrides.items())) if context_overrides else None,
        preserve_trial_context=True,
        note_types=args.note_types,
        evidence_sha256=file_sha256(evidence_path),
        evidence_schema_version=schema_version,
        cohort_mrn_count=len(current_cohort) if current_cohort is not None else None,
        # The denominator itself, so the runner can emit explicit rows for cohort
        # patients who never produced a trigger-bearing note.
        cohort_mrns=current_cohort,
    )
    progress.update(1)
    progress.close()
    summarize_evidence(evidence, label)
    print(f"Wrote {label} evidence ({evidence.height} notes): {evidence_path}")
    return evidence_path


def add_collector_args(parser, *, default_output_dir, default_mrn_file, default_context_chars):
    """CLI flags shared by the per-note collectors."""
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=default_mrn_file,
        help="CSV cohort file containing DFCI_MRN values; defaults to the "
             "COMPASS ICD prostate cohort.",
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
        default=default_context_chars,
        help="Chars of context kept on each side of a trigger match.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=None,
        help="Processes for note cleaning and trigger scanning (default: all cores).",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescan from scratch even if existing evidence matches these settings.",
    )
    return parser
