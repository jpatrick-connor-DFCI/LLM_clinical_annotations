"""Stage 1 — Scan clinical notes for stage mentions and write an evidence table.

For every patient, notes are scanned for stage triggers, context windows are
extracted around each match, copy-forward notes are de-duplicated per patient,
and the resulting snippets are written to a TSV evidence table. This step runs
before any LLM calls so the scanning layer can be audited and re-used independently.

Default source: the full OncDRS raw text corpus (no pre-specified MRN list required).
Raw file scanning is parallelised over files using ProcessPoolExecutor.

Incremental output (raw file path only):
  Snippets are written to stage_evidence_raw.tsv as each file completes. Processed
  files are logged to stage_scanned_files.tsv. Re-running without --overwrite resumes
  from where the scan left off. stage_evidence.tsv is always rebuilt from the raw TSV
  at the end via a dedup pass.

Outputs (under <output-dir>):
  stage_evidence.tsv         Deduped snippets — one row per unique (patient, snippet).
  stage_evidence_raw.tsv     Pre-dedup snippets, written incrementally (parallel path).
  stage_scanned_files.tsv    Per-file scan log used for resumability (parallel path).

Usage:
  python cancer_stage/extract_stage_notes.py --output-dir /path/to/output
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.longitudinal_helpers import (  # noqa: E402
    find_matches,
    iter_note_snippets,
    load_selected_mrns,
)
from shared.llm_helpers import (  # noqa: E402
    DEFAULT_RAW_TEXT_PATHS,
    build_raw_note_row,
    build_snippet,
    clean_note,
    discover_raw_text_files,
    extract_raw_docs,
    load_note_bundle,
    load_notes_csv,
    load_raw_text_notes,
    to_iso_date,
)

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("STAGE_OUTPUT_DIR", "/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/")
)

_ONCDRS_ROOT = Path("/data/gusev/PROFILE/CLINICAL/OncDRS/")
DEFAULT_STAGE_RAW_TEXT_PATHS = (
    *DEFAULT_RAW_TEXT_PATHS,
    _ONCDRS_ROOT / "CLINICAL_TEXTS_2026_03",
)

STAGE_TRIGGER_REGEX = {
    "stage_group": (
        # "clinical stage IV", "pathologic stage IIIA", "Stage 2b", "stage four"
        r"\b(?:clinical|pathologic|pathological)\s+stage\s+"
        r"(?:IV[ABCabc]?|III[ABCabc]?|II[ABCabc]?|I[ABCabc]?|[0-4][ABCabc]?)\b"
        r"|\bstage\s+"
        r"(?:IV[ABCabc]?|III[ABCabc]?|II[ABCabc]?|I[ABCabc]?|[0-4][ABCabc]?)\b"
        r"|\bstage\s+(?:one|two|three|four)\b"
    ),
    "staging_system": r"\b(?:AJCC|FIGO|Ann\s+Arbor)\b",
    "limited_extensive": r"\b(?:limited|extensive)\s+stage\b",
}

EVIDENCE_COLUMNS = ["note_uid", "DFCI_MRN", "note_date", "note_type", "trigger_categories", "snippet"]
SCANNED_COLUMNS = ["file_path", "note_type", "n_snippets", "status"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _n_workers(cap=8):
    """Cores allocated to this job (SLURM-aware), capped for memory safety.

    os.cpu_count() reports all physical cores and ignores SLURM cgroup limits,
    which causes oversubscription on shared nodes. sched_getaffinity reads the
    actual CPU allocation.
    """
    try:
        allocated = len(os.sched_getaffinity(0))
    except AttributeError:
        allocated = os.cpu_count() or 1
    return max(1, min(cap, allocated))


def _note_uid(mrn, note_date, snippet, raw_note_id):
    if raw_note_id is not None:
        return str(raw_note_id)
    key = f"{int(mrn)}|{note_date or ''}|{snippet[:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _append_rows(path, rows, columns):
    """Append rows to a TSV, writing the header only on the first write."""
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        mode="a",
        sep="\t",
        index=False,
        header=not path.exists() or path.stat().st_size == 0,
    )


def _records_to_tsv_rows(records):
    """Convert snippet dicts (trigger_categories as list) to TSV-ready dicts."""
    return [
        {
            "note_uid": r["note_uid"],
            "DFCI_MRN": r["DFCI_MRN"],
            "note_date": r["note_date"],
            "note_type": r["note_type"],
            "trigger_categories": ",".join(r["trigger_categories"]),
            "snippet": r["snippet"],
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# Per-file worker (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _scan_file(args):
    """Load one JSON file, find stage matches, return snippet records.

    STAGE_TRIGGER_REGEX is resolved at import time in each worker process.
    """
    file_path, note_type, context_chars = args
    rows = []
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for note in extract_raw_docs(payload):
        row = build_raw_note_row(note, note_type, file_path)
        if row is None:
            continue
        mrn = row["DFCI_MRN"]
        cleaned = clean_note(row["CLINICAL_TEXT"], note_type=note_type)
        if not cleaned:
            continue
        matches = find_matches(cleaned, STAGE_TRIGGER_REGEX)
        if not matches:
            continue
        snippet = build_snippet(cleaned, matches, context_chars=context_chars)
        if not snippet:
            continue
        note_date = to_iso_date(row.get("EVENT_DATE"))
        rows.append({
            "note_uid": _note_uid(mrn, note_date, snippet, row.get("RAW_NOTE_ID")),
            "DFCI_MRN": int(mrn),
            "note_date": note_date,
            "note_type": note_type,
            "trigger_categories": sorted({m[0] for m in matches}),
            "snippet": snippet,
        })
    return rows


# ---------------------------------------------------------------------------
# Parallel raw-file scan with incremental output
# ---------------------------------------------------------------------------

def _parallel_scan_incremental(raw_text_paths, context_chars, max_workers, raw_path, scanned_path):
    """Scan raw JSON files in parallel, writing results incrementally.

    Already-scanned files (present in scanned_path with status "ok") are skipped,
    so a re-run without --overwrite resumes from where the previous scan stopped.
    Results are appended to raw_path as each future completes.
    """
    raw_files = discover_raw_text_files(raw_text_paths)
    if not raw_files:
        joined = ", ".join(str(p) for p in raw_text_paths)
        raise FileNotFoundError(f"No supported raw JSON files found under: {joined}")

    done = set()
    if scanned_path.exists() and scanned_path.stat().st_size > 0:
        scanned_df = pd.read_csv(scanned_path, sep="\t", dtype=str)
        done = set(scanned_df.loc[scanned_df["status"] == "ok", "file_path"])

    todo = [(fp, nt) for fp, nt in raw_files if str(fp) not in done]
    print(
        f"Files: {len(raw_files)} total, {len(done)} already scanned, "
        f"{len(todo)} remaining"
    )
    if not todo:
        return

    args_list = [(fp, nt, context_chars) for fp, nt in todo]
    total_snippets = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_file, a): a for a in args_list}
        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Scanning files ({max_workers} workers)",
            unit="file",
        )
        for future in bar:
            file_path, note_type, _ = futures[future]
            try:
                records = future.result()
                _append_rows(raw_path, _records_to_tsv_rows(records), EVIDENCE_COLUMNS)
                _append_rows(
                    scanned_path,
                    [{"file_path": str(file_path), "note_type": note_type,
                      "n_snippets": len(records), "status": "ok"}],
                    SCANNED_COLUMNS,
                )
                total_snippets += len(records)
            except Exception as exc:
                errors += 1
                _append_rows(
                    scanned_path,
                    [{"file_path": str(file_path), "note_type": note_type,
                      "n_snippets": 0, "status": f"error: {exc!r}"}],
                    SCANNED_COLUMNS,
                )
                print(f"\nWarning: skipped {file_path}: {exc!r}", file=sys.stderr)
            bar.set_postfix(snippets=total_snippets, errors=errors, refresh=False)

    if errors:
        print(f"Scan complete with {errors} file error(s). Check stderr for details.")


def _build_evidence_from_raw(raw_path, evidence_path, note_types=None):
    """Dedup stage_evidence_raw.tsv into stage_evidence.tsv.

    Applies optional note-type filter, then deduplicates on (DFCI_MRN, snippet)
    keeping the earliest note_date for each unique pair.
    """
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pd.DataFrame(columns=EVIDENCE_COLUMNS).to_csv(evidence_path, sep="\t", index=False)
        return 0

    raw_df = pd.read_csv(raw_path, sep="\t", dtype=str, on_bad_lines="warn")

    if note_types:
        wanted = {t.strip().lower() for t in note_types}
        raw_df = raw_df[raw_df["note_type"].str.lower().isin(wanted)]
        print(f"After note-type filter {list(note_types)}: {len(raw_df)} raw snippets")

    # Sort so keep="first" in drop_duplicates retains the earliest note_date.
    raw_df["_date_sort"] = pd.to_datetime(raw_df["note_date"], errors="coerce")
    raw_df = raw_df.sort_values("_date_sort", na_position="last").drop(columns=["_date_sort"])

    deduped = raw_df.drop_duplicates(subset=["DFCI_MRN", "snippet"], keep="first")
    deduped = deduped.sort_values(["DFCI_MRN", "note_date"], na_position="last")
    deduped.to_csv(evidence_path, sep="\t", index=False)
    return len(deduped)


# ---------------------------------------------------------------------------
# Sequential path (CSV / bundle / MRN-filtered raw text)
# ---------------------------------------------------------------------------

def _load_and_scan_sequential(args, selected_mrns, context_chars):
    """Load notes from a non-default source and scan sequentially."""
    if args.note_bundle_path is not None:
        notes_df = load_note_bundle(args.note_bundle_path, selected_mrns)
        print(f"Loaded bundle: {len(notes_df)} rows for {notes_df['DFCI_MRN'].nunique()} patients")
    elif args.notes_csv is not None:
        notes_df = load_notes_csv(args.notes_csv, selected_mrns)
        print(f"Loaded CSV: {len(notes_df)} rows for {notes_df['DFCI_MRN'].nunique()} patients")
    else:
        raw_paths = args.raw_text_path or list(DEFAULT_STAGE_RAW_TEXT_PATHS)
        notes_df = load_raw_text_notes(raw_paths, selected_mrns)
        print(f"Loaded {len(notes_df)} rows for {notes_df['DFCI_MRN'].nunique()} patients")
    return list(iter_note_snippets(notes_df, STAGE_TRIGGER_REGEX, context_chars=context_chars))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan clinical notes for stage mentions and write a snippet evidence table. "
                    "By default scans the full OncDRS raw text corpus without an MRN filter."
    )
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="Optional: restrict scan to these MRNs "
                             "(file with one MRN per line, or CSV with DFCI_MRN column).")
    parser.add_argument("--mrns", default=None,
                        help="Optional: comma- or space-separated MRNs to restrict the scan.")
    parser.add_argument("--notes-csv", type=Path, default=None,
                        help="Optional: load notes from a pre-compiled CSV instead of raw files.")
    parser.add_argument("--note-bundle-path", type=Path, default=None,
                        help="Optional: load notes from a .json.gz bundle.")
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None,
                        help="Optional: raw OncDRS JSON directory. Repeat to add multiple. "
                             f"Default: {[str(p) for p in DEFAULT_STAGE_RAW_TEXT_PATHS]}")
    parser.add_argument("--note-types", nargs="+", default=None,
                        help="Optional: restrict to these NOTE_TYPE values "
                             "(e.g. Pathology Clinician). Default: all note types.")
    parser.add_argument("--context-chars", type=int, default=600,
                        help="Characters of context on each side of a trigger match (default: 600).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Parallel workers for raw file scanning. "
                             "Default: SLURM-allocated cores, capped at 8.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear all existing output files and rescan from scratch.")
    return parser.parse_args()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "stage_evidence.tsv"
    raw_path = args.output_dir / "stage_evidence_raw.tsv"
    scanned_path = args.output_dir / "stage_scanned_files.tsv"

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    max_workers = args.max_workers or _n_workers()

    # Parallel incremental path: full corpus scan, no pre-specified source or MRN list.
    is_parallel_path = (
        args.note_bundle_path is None
        and args.notes_csv is None
        and selected_mrns is None
    )

    if args.overwrite:
        evidence_path.unlink(missing_ok=True)
        if is_parallel_path:
            raw_path.unlink(missing_ok=True)
            scanned_path.unlink(missing_ok=True)

    if is_parallel_path:
        raw_paths = args.raw_text_path or list(DEFAULT_STAGE_RAW_TEXT_PATHS)
        _parallel_scan_incremental(
            raw_paths, args.context_chars, max_workers, raw_path, scanned_path
        )
        n = _build_evidence_from_raw(raw_path, evidence_path, args.note_types)
    else:
        records = _load_and_scan_sequential(args, selected_mrns, args.context_chars)
        if args.note_types:
            wanted = {t.strip().lower() for t in args.note_types}
            before = len(records)
            records = [r for r in records if (r["note_type"] or "").lower() in wanted]
            print(f"After note-type filter {args.note_types}: {len(records)}/{before} snippets")
        evidence_df = pd.DataFrame(_records_to_tsv_rows(records), columns=EVIDENCE_COLUMNS)
        if not evidence_df.empty:
            evidence_df = evidence_df.sort_values(["DFCI_MRN", "note_date"], na_position="last")
        evidence_df.to_csv(evidence_path, sep="\t", index=False)
        n = len(evidence_df)

    n_patients = 0
    if evidence_path.exists() and evidence_path.stat().st_size > 0:
        n_patients = pd.read_csv(evidence_path, sep="\t", usecols=["DFCI_MRN"])["DFCI_MRN"].nunique()
    print(f"Wrote {n} evidence snippets for {n_patients} patients: {evidence_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
