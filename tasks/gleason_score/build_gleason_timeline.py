"""Stage 2 — Call the LLM on per-patient Gleason evidence chunks; write the timeline.

Reads gleason_evidence.tsv produced by collect_gleason_notes.py. Calls the LLM
once per chunk, and aggregates the findings into a deduped per-patient timeline.

Outputs (under <output-dir>):
  gleason_extractions_raw.tsv     per-finding extractions (provenance, pre-dedup)
  gleason_processed_chunks.tsv    per-chunk log — the unit of resume. Each row
                                   also records the evidence scan_config hash
                                   (read from gleason_evidence.meta.json) it was
                                   produced under, so a regenerated evidence file
                                   with different scan params can't silently
                                   "resume" onto now-mismatched chunk indices.
  gleason_processed_patients.tsv  processed-patient log (derived per-patient status)
  gleason_timeline.tsv            deduped timeline (every score + date per patient)

Evidence-hash guard: if gleason_evidence.meta.json exists (written by
collect_gleason_notes.py) and its scan_config disagrees with the value already
recorded in gleason_processed_chunks.tsv, this raises rather than resuming —
the evidence was regenerated with different scan parameters, so old chunk
indices no longer mean the same thing. Restore the matching evidence file or
re-run with --overwrite to discard the stale chunk log. If no meta sidecar
exists (evidence generated before this check), this warns and proceeds, since
older evidence on disk has no recorded hash to compare against.

Usage:
  # Run collection first:
  python preprocessing/cli/collect_gleason_notes.py --output-dir /path/to/output

  # Then run LLM extraction:
  python tasks/gleason_score/build_gleason_timeline.py --output-dir /path/to/output
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import CLINICAL_SAFETY_CONTEXT, DEFAULT_DATA_PATH  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    derive_grade_group,
    flatten_ws,
    read_scan_config_meta,
    resolve_date,
)
from preprocessing.notes import load_selected_mrns  # noqa: E402
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.gleason_score.prompts import GLEASON_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_gleason_timeline"

RAW_COLUMNS = [
    "DFCI_MRN",
    "chunk_index",
    "source_note_date",
    "gleason_primary",
    "gleason_secondary",
    "gleason_total",
    "grade_group",
    "specimen_type",
    "scoring_date",
    "is_historical_reference",
    "quote",
]

# Sanity bound on chunk_index read back from the evidence TSV. Real patients have
# single-digit chunk counts; anything beyond this is a misaligned row.
MAX_CHUNK_INDEX = 10_000

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "gleason_date",
    "date_source",
    "gleason_primary",
    "gleason_secondary",
    "gleason_total",
    "grade_group",
    "specimen_type",
    "is_historical_reference",
    "supporting_quote",
    "source_note_date",
]

PROCESSED_COLUMNS = [
    "DFCI_MRN",
    "num_chunks",
    "num_chunks_ok",
    "num_findings",
    "status",
]

# Per-chunk log: the unit of resume. A chunk that failed is retried on the next
# run without re-calling the chunks that already succeeded. scan_config records
# the evidence hash (from gleason_evidence.meta.json) each row's chunk_index
# was assigned under, so a regenerated evidence file with different scan params
# can be detected before "resuming" onto now-mismatched chunks.
CHUNK_COLUMNS = ["DFCI_MRN", "chunk_index", "num_findings", "status", "scan_config"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the LLM on Gleason evidence chunks and write a Gleason timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing gleason_evidence.tsv and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help="Override path to gleason_evidence.tsv.")
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument(
        "--provider",
        choices=["dfci_gpt", "vertex_ai"],
        default="dfci_gpt",
        help="Which LLM backend to call.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to the selected provider's default_model.",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_rows(path, rows, columns):
    """Append rows to a TSV, writing the header only on the first write.

    Polars has no append mode for write_csv, so the CSV text is generated
    in-memory and appended via a plain file handle.
    """
    if not rows:
        return
    df = pl.DataFrame({c: [r.get(c) for r in rows] for c in columns})
    write_header = not path.exists() or path.stat().st_size == 0
    text = df.write_csv(separator="\t", include_header=write_header)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def read_done_chunks(path):
    """Return {(mrn, chunk_index)} for every chunk logged as status == "ok"."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    log = pl.read_csv(
        path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True
    )
    if "DFCI_MRN" not in log.columns or "chunk_index" not in log.columns:
        return set()
    done = set()
    for row in log.iter_rows(named=True):
        if row.get("status") != "ok":
            continue
        mrn = _to_int(row.get("DFCI_MRN"))
        idx = _to_int(row.get("chunk_index"))
        if mrn is None or idx is None:
            continue
        done.add((mrn, idx))
    return done


def check_scan_config(chunk_log_path, meta_path):
    """Guard chunk-index resume against a regenerated evidence file.

    Compares the scan_config hash recorded in the existing chunk log (if any)
    against the hash recorded in the evidence meta sidecar (if any):

    - Both present, mismatched -> raise. The evidence was regenerated with
      different scan parameters, so old chunk_index values in the log no
      longer correspond to the same snippets; resuming would silently extract
      the wrong text. Restore the matching evidence file, or re-run with
      --overwrite to discard the stale chunk log and start clean.
    - No meta sidecar (evidence predates this check) -> warn and proceed. This
      is the legacy path: there is nothing to validate against, so it is not
      treated as an error.
    - No existing chunk log (first run for this output dir) -> nothing to
      check.

    Returns the current scan_config to record on new chunk-log rows (or None
    if there is no meta sidecar to record).
    """
    meta = read_scan_config_meta(meta_path)
    current_config = meta.get("scan_config") if meta else None

    if not chunk_log_path.exists() or chunk_log_path.stat().st_size == 0:
        if current_config is None:
            print(
                f"Warning: no scan-config sidecar found at {meta_path} "
                "(evidence predates scan-config tracking); proceeding without "
                "a hash guard."
            )
        return current_config

    log = pl.read_csv(
        chunk_log_path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True
    )
    if "scan_config" not in log.columns:
        print(
            f"Warning: {chunk_log_path} predates scan-config tracking; proceeding "
            "without a hash guard. Re-run with --overwrite if you suspect the "
            "evidence file has changed since this chunk log was built."
        )
        return current_config

    recorded_configs = set(log["scan_config"].drop_nulls().cast(pl.Utf8).to_list())
    if not recorded_configs:
        return current_config

    if current_config is None:
        print(
            f"Warning: no scan-config sidecar found at {meta_path}, but the "
            f"existing chunk log at {chunk_log_path} was built under a recorded "
            "hash. Proceeding without a hash guard — restore the evidence meta "
            "sidecar if you want this checked."
        )
        return current_config

    if recorded_configs != {current_config}:
        raise ValueError(
            "Evidence scan settings differ from the existing chunk log "
            f"({sorted(recorded_configs)} != [{current_config}]). chunk_index "
            "values in the existing log no longer match this evidence file. "
            "Restore the matching evidence file, or re-run with --overwrite "
            "to discard the stale chunk log and reprocess from scratch."
        )
    return current_config


def compact_log(path, columns, key_columns):
    """Rewrite an append-only log keeping only the LAST row per key.

    Retries append a fresh row rather than rewriting in place (which keeps the
    hot loop crash-safe), so a patient retried across runs accumulates one row
    per attempt. Collapsing at the end keeps the log a clean current-state view.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    log = pl.read_csv(
        path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True
    )
    if not all(c in log.columns for c in key_columns):
        return
    before = log.height
    compacted = log.unique(subset=key_columns, keep="last", maintain_order=True)
    if compacted.height == before:
        return
    tmp_path = path.with_name(f".{path.name}.tmp")
    compacted.select([c for c in columns if c in compacted.columns]).write_csv(
        tmp_path, separator="\t"
    )
    tmp_path.replace(path)
    print(f"  Compacted {path.name}: {before} -> {compacted.height} rows")


def dedupe_raw_findings(path, columns, key_columns):
    """Drop superseded rows from the raw findings log after a chunk retry.

    Unlike compact_log, `.unique(keep="last")` is wrong here: a single chunk
    legitimately writes MANY rows sharing one (mrn, chunk_index) — one per
    finding — so collapsing to one row per key would destroy real findings,
    not just retry duplicates. Instead, treat each contiguous run of rows
    sharing a key as one "occurrence" (the hot loop appends a chunk's rows
    together, so a retry's rows form a later, separate run) and keep every
    row in the LAST occurrence, dropping earlier occurrences whole.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    log = pl.read_csv(
        path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True
    )
    if not all(c in log.columns for c in key_columns):
        return
    before = log.height
    key = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in key_columns], separator="\x1f")
    log = log.with_columns(key.alias("_key"))
    # New run id each time the key differs from the previous row (rows within
    # one chunk's append are contiguous, so this separates attempt from retry).
    new_run = (log["_key"] != log["_key"].shift(1)).fill_null(True)
    log = log.with_columns(new_run.cum_sum().alias("_run"))
    last_run = log.group_by("_key").agg(pl.col("_run").max().alias("_last_run"))
    log = log.join(last_run, on="_key")
    deduped = log.filter(pl.col("_run") == pl.col("_last_run"))
    if deduped.height == before:
        return
    tmp_path = path.with_name(f".{path.name}.tmp")
    deduped.select([c for c in columns if c in deduped.columns]).write_csv(
        tmp_path, separator="\t"
    )
    tmp_path.replace(path)
    print(f"  Deduped {path.name}: {before} -> {deduped.height} rows")


def _extract_chunk(provider, client, model, max_retries, mrn, chunk):
    """Run one LLM call for a single chunk.

    Returns (findings, error). Exactly one of the two is meaningful: on error
    findings is None.
    """
    payload = {
        "patient_mrn": int(mrn),
        "notes": [
            {"note_date": r["note_date"], "note_type": r["note_type"], "note_text": r["snippet"]}
            for r in chunk
        ],
    }
    messages = [
        {"role": "system", "content": GLEASON_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response_text, error = provider.call_with_retry(client, model, messages, max_retries)
    if error:
        return None, error
    try:
        result = parse_json_response(response_text)
    except json.JSONDecodeError as exc:
        return None, f"json_parse: {exc}"
    if not isinstance(result, dict):
        return None, f"non_dict_response: {type(result).__name__}"
    found = result.get("gleason_findings")
    if not isinstance(found, list):
        return None, "missing_gleason_findings"
    return [f for f in found if isinstance(f, dict)], None


def extract_patient(provider, client, model, max_retries, mrn, indexed_chunks):
    """Run one LLM call per chunk, keeping the findings from every chunk that works.

    `indexed_chunks` is a list of (chunk_index, chunk) pairs — only the chunks
    still outstanding for this patient, so a resumed run never re-calls a chunk
    that already succeeded.

    Returns (findings, chunk_results):
      findings      [(finding_dict, chunk_index), ...] for every chunk that
                    succeeded. A failing chunk never discards its siblings' work.
      chunk_results [{"chunk_index", "num_findings", "status"}, ...], one per
                    attempted chunk, where status is "ok" or the error string.
    """
    findings = []
    chunk_results = []
    for chunk_index, chunk in indexed_chunks:
        chunk_findings, error = _extract_chunk(
            provider, client, model, max_retries, mrn, chunk
        )
        if error:
            chunk_results.append(
                {"chunk_index": chunk_index, "num_findings": 0, "status": error}
            )
            continue
        findings.extend((f, chunk_index) for f in chunk_findings)
        chunk_results.append({
            "chunk_index": chunk_index,
            "num_findings": len(chunk_findings),
            "status": "ok",
        })
    return findings, chunk_results


def raw_rows_from_findings(mrn, findings):
    rows = []
    for finding, chunk_index in findings:
        rows.append({
            "DFCI_MRN": int(mrn),
            "chunk_index": chunk_index,
            "source_note_date": finding.get("source_note_date"),
            "gleason_primary": finding.get("primary"),
            "gleason_secondary": finding.get("secondary"),
            "gleason_total": finding.get("total"),
            "grade_group": finding.get("grade_group"),
            "specimen_type": finding.get("specimen_type"),
            "scoring_date": finding.get("scoring_date"),
            "is_historical_reference": finding.get("is_historical_reference"),
            "quote": flatten_ws(finding.get("quote")),
        })
    return rows


def _to_int(value):
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_timeline(raw_path, timeline_path):
    """Resolve dates, validate, and de-duplicate raw extractions into the timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pl.DataFrame(schema={c: pl.Utf8 for c in TIMELINE_COLUMNS}).write_csv(
            timeline_path, separator="\t"
        )
        return 0

    # Read every field as text and validate per row, so a single malformed/misaligned
    # row (e.g. free-text that shifted columns) can't abort the whole timeline build.
    raw = pl.read_csv(raw_path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)
    seen = set()
    rows = []
    skipped = 0
    # Findings that parse fine but fail the clinical-range validation below (bad
    # total, or primary/secondary outside 1-5) are otherwise dropped silently,
    # hiding model errors behind a lower row count; count them so they're visible.
    invalid_score = 0
    for r in raw.iter_rows(named=True):
        mrn_val = _to_int(r.get("DFCI_MRN"))
        if mrn_val is None:
            skipped += 1
            continue
        mrn = mrn_val

        primary = _to_int(r.get("gleason_primary"))
        secondary = _to_int(r.get("gleason_secondary"))
        total = _to_int(r.get("gleason_total"))
        # Gleason total is primary + secondary by definition; recompute it when
        # both patterns are present so an LLM arithmetic slip can't propagate.
        if primary is not None and secondary is not None:
            total = primary + secondary
        # Require a usable total; drop grade-group-only or malformed extractions.
        if total is None or not (2 <= total <= 10):
            invalid_score += 1
            continue
        if primary is not None and not (1 <= primary <= 5):
            invalid_score += 1
            continue
        if secondary is not None and not (1 <= secondary <= 5):
            invalid_score += 1
            continue

        grade_group = _to_int(r.get("grade_group"))
        if grade_group is None or not (1 <= grade_group <= 5):
            grade_group = derive_grade_group(primary, secondary)

        gleason_date, date_source = resolve_date(
            r.get("scoring_date"), r.get("source_note_date")
        )
        specimen_type = r.get("specimen_type")

        key = (mrn, primary, secondary, total, gleason_date, specimen_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "DFCI_MRN": mrn,
            "gleason_date": gleason_date,
            "date_source": date_source,
            "gleason_primary": primary,
            "gleason_secondary": secondary,
            "gleason_total": total,
            "grade_group": grade_group,
            "specimen_type": specimen_type,
            "is_historical_reference": r.get("is_historical_reference"),
            "supporting_quote": r.get("quote"),
            "source_note_date": r.get("source_note_date"),
        })

    if skipped:
        print(f"  Skipped {skipped} malformed/misaligned raw rows during timeline build")
    if invalid_score:
        print(f"  Dropped {invalid_score} findings with an invalid/out-of-range Gleason score")

    if not rows:
        timeline = pl.DataFrame(schema={c: pl.Utf8 for c in TIMELINE_COLUMNS})
    else:
        timeline = pl.DataFrame({c: [row.get(c) for row in rows] for c in TIMELINE_COLUMNS})
        # Nullable Int64 so integer grades render as "3"/"", not "3.0"/"NaN".
        int_cols = ["gleason_primary", "gleason_secondary", "gleason_total", "grade_group"]
        timeline = timeline.with_columns(
            [pl.col(c).cast(pl.Int64, strict=False) for c in int_cols]
        )
        timeline = timeline.sort(
            ["DFCI_MRN", "gleason_date"], nulls_last=True
        )
    timeline.write_csv(timeline_path, separator="\t")
    return timeline.height


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "gleason_evidence.tsv")
    meta_path = args.output_dir / "gleason_evidence.meta.json"
    raw_path = args.output_dir / "gleason_extractions_raw.tsv"
    chunk_log_path = args.output_dir / "gleason_processed_chunks.tsv"
    processed_path = args.output_dir / "gleason_processed_patients.tsv"
    timeline_path = args.output_dir / "gleason_timeline.tsv"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_gleason_notes.py first."
        )

    if args.overwrite:
        for path in (raw_path, chunk_log_path, processed_path, timeline_path):
            path.unlink(missing_ok=True)

    # Must run BEFORE read_done_chunks: raises if the evidence file was
    # regenerated with different scan params than the existing chunk log was
    # built under (chunk_index would then mean something different), warns and
    # proceeds if the evidence predates scan-config tracking (legacy data).
    scan_config = check_scan_config(chunk_log_path, meta_path)

    evidence_df = pl.read_csv(evidence_path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)
    evidence_df = evidence_df.with_columns(
        pl.col("DFCI_MRN").cast(pl.Float64, strict=False).alias("DFCI_MRN")
    ).drop_nulls(subset=["DFCI_MRN"]).with_columns(
        pl.col("DFCI_MRN").cast(pl.Int64)
    )
    print(
        f"Loaded evidence: {evidence_df.height} snippets for "
        f"{evidence_df['DFCI_MRN'].n_unique()} patients"
    )

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    if selected_mrns is not None:
        evidence_df = evidence_df.filter(pl.col("DFCI_MRN").is_in(selected_mrns))
        print(f"After MRN filter: {evidence_df.height} snippets for "
              f"{evidence_df['DFCI_MRN'].n_unique()} patients")

    patient_chunks = {}
    bad_chunk_index = 0
    for row in evidence_df.iter_rows(named=True):
        mrn = int(row["DFCI_MRN"])
        rec = {
            "note_date": row.get("note_date"),
            "note_type": row.get("note_type") or "Unknown",
            "snippet": row.get("snippet") or "",
        }
        try:
            chunk_index = int(row.get("chunk_index") or 0)
        except (TypeError, ValueError):
            bad_chunk_index += 1
            continue
        # A misaligned row (which truncate_ragged_lines lets through) can carry a
        # nonsense chunk_index; without a bound the fill loop below would allocate
        # that many empty lists.
        if not 0 <= chunk_index <= MAX_CHUNK_INDEX:
            bad_chunk_index += 1
            continue
        chunks = patient_chunks.setdefault(mrn, [])
        while len(chunks) <= chunk_index:
            chunks.append([])
        chunks[chunk_index].append(rec)

    if bad_chunk_index:
        print(f"  Skipped {bad_chunk_index} evidence rows with an invalid chunk_index")

    # Chunks are packed contiguously by collect_gleason_notes.py, but a filtered or
    # partially-malformed evidence file can leave a hole; drop empties so they
    # aren't dispatched as empty LLM calls.
    patient_chunks = {
        mrn: [c for c in chunks if c]
        for mrn, chunks in patient_chunks.items()
    }
    patient_chunks = {mrn: chunks for mrn, chunks in patient_chunks.items() if chunks}

    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients to process: {len(patient_chunks)} "
        f"({total_chunks} LLM calls across chunks)"
    )

    # Resume at chunk granularity: a patient whose chunk 2 failed re-runs only
    # chunk 2, keeping the findings chunks 0 and 1 already produced.
    done_chunks = read_done_chunks(chunk_log_path)
    if done_chunks:
        print(f"Already completed chunks: {len(done_chunks)}")

    todo = []
    for mrn in sorted(patient_chunks):
        outstanding = [
            (i, chunk)
            for i, chunk in enumerate(patient_chunks[mrn])
            if (mrn, i) not in done_chunks
        ]
        if outstanding:
            todo.append((mrn, outstanding))
    if args.limit_patients is not None:
        todo = todo[: args.limit_patients]
    outstanding_chunks = sum(len(c) for _, c in todo)
    print(
        f"Patients to extract with LLM: {len(todo)} "
        f"({outstanding_chunks} outstanding chunks)"
    )

    if todo:
        provider = get_provider(args.provider)
        model = args.model or provider.default_model
        client = provider.build_client()

        def worker(mrn, indexed_chunks):
            findings, chunk_results = extract_patient(
                provider, client, model, args.max_retries, mrn, indexed_chunks
            )
            return mrn, findings, chunk_results

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(worker, mrn, indexed_chunks): mrn
                for mrn, indexed_chunks in todo
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Patients", unit="pt"
            ):
                mrn, findings, chunk_results = future.result()
                rows = raw_rows_from_findings(mrn, findings)
                append_rows(raw_path, rows, RAW_COLUMNS)
                append_rows(
                    chunk_log_path,
                    [{"DFCI_MRN": int(mrn), "scan_config": scan_config, **r} for r in chunk_results],
                    CHUNK_COLUMNS,
                )
                n_total = len(patient_chunks[mrn])
                n_ok = sum(1 for r in chunk_results if r["status"] == "ok")
                # Add back chunks skipped by resume (never in chunk_results this run).
                # Valid ONLY because read_done_chunks filters to status == "ok", so
                # "unattempted" and "already ok" are the same set today. If a future
                # skip reason is added (e.g. an evidence-hash mismatch invalidating
                # stale chunk indices), that equivalence breaks and this needs to
                # explicitly track ok-vs-skipped-for-other-reasons separately.
                n_ok += n_total - len(chunk_results)  # chunks done on an earlier run
                failed = [r["status"] for r in chunk_results if r["status"] != "ok"]
                if not failed:
                    status = "ok"
                elif n_ok:
                    status = f"partial:{len(failed)}/{n_total}"
                else:
                    status = f"failed:{failed[0]}"
                append_rows(
                    processed_path,
                    [{
                        "DFCI_MRN": int(mrn),
                        "num_chunks": n_total,
                        "num_chunks_ok": n_ok,
                        "num_findings": len(rows),
                        "status": status,
                    }],
                    PROCESSED_COLUMNS,
                )

    # Compact once at the end rather than per patient: the hot loop stays
    # append-only (crash-safe), and retried patients leave one row, not one per run.
    compact_log(processed_path, PROCESSED_COLUMNS, ["DFCI_MRN"])
    compact_log(chunk_log_path, CHUNK_COLUMNS, ["DFCI_MRN", "chunk_index"])
    # Raw findings need a different collapse than the two logs above: a retried
    # chunk re-appends its findings, and without this they'd double-count in the
    # timeline build (build_timeline dedupes by score/date/specimen, so exact
    # duplicates collapse there, but a retry whose finding set genuinely CHANGED
    # would otherwise double-count instead of superseding).
    dedupe_raw_findings(raw_path, RAW_COLUMNS, ["DFCI_MRN", "chunk_index"])

    n = build_timeline(raw_path, timeline_path)
    print(f"Wrote Gleason timeline ({n} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
