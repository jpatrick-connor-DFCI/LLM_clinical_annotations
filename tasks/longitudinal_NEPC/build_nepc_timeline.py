"""Stage 2 — Call the LLM on per-patient AVPC/NEPC evidence chunks; write the timeline.

Reads avpc_nepc_evidence.tsv produced by collect_nepc_notes.py. Calls the LLM once
per chunk, recording which Aparicio aggressive-variant criteria (C1-C7) and which
NEPC sub-features are documented as present, with the date each was diagnosed.
Per-call extractions are aggregated into a per-patient onset timeline: one row each
time a NEW criterion is first added to the patient's record, carrying the cumulative
set of criteria to that date.

Outputs (under <output-dir>):
  avpc_nepc_extractions_raw.tsv   per-finding extractions (provenance, pre-aggregation)
  avpc_nepc_processed_chunks.tsv  per-chunk log — the unit of resume
  avpc_nepc_processed_patients.tsv  processed-patient log (derived per-patient status)
  avpc_nepc_timeline.tsv          one row per newly-added criterion (with cumulative set)

Usage:
  # Run collection first:
  python preprocessing/cli/collect_nepc_notes.py --output-dir /path/to/output

  # Then run LLM extraction:
  python tasks/longitudinal_NEPC/build_nepc_timeline.py --output-dir /path/to/output
"""

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import CLINICAL_SAFETY_CONTEXT, DEFAULT_DATA_PATH  # noqa: E402
from preprocessing.longitudinal import flatten_ws, resolve_date  # noqa: E402
from preprocessing.notes import load_selected_mrns  # noqa: E402
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.longitudinal_NEPC.prompts import NEPC_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_avpc_nepc_timeline"

CRITERION_LABELS = {
    "C1": "small-cell histology",
    "C2": "visceral metastatic pattern (lung/adrenal/brain/pleura/peritoneum)",
    "C3": "predominantly lytic bone metastases",
    "C4": "bulky disease (bulky nodal or prostate/pelvic mass >= 5 cm)",
    "C5": "low PSA with high-volume disease",
    "C6": "neuroendocrine markers / elevated CEA or LDH / hypercalcemia",
    "C7": "rapid progression to castration-resistant / androgen-independent disease",
    "NEPC:small_cell_dx": "NEPC: neuroendocrine or small-cell carcinoma diagnosis",
    "NEPC:histologic_transformation": "NEPC: histologic transformation from adenocarcinoma",
    "NEPC:ne_features": "NEPC: neuroendocrine features / differentiation",
    "NEPC:positive_ne_ihc": "NEPC: positive neuroendocrine IHC (synaptophysin/chromogranin/CD56/NSE/INSM1)",
}
VALID_CRITERIA = set(CRITERION_LABELS)
VISCERAL_PATTERNS = {"visceral_only", "visceral_and_bone", "none"}

RAW_COLUMNS = [
    "DFCI_MRN",
    "chunk_index",
    "source_note_date",
    "criterion",
    "diagnosis_date",
    "modality",
    "visceral_met_pattern",
    "quote",
    "confidence",
]

# Verbatim quotes are free text from the model; cap them so an over-long quote
# can't produce the misaligned raw rows build_timeline already counts as skipped.
MAX_QUOTE_CHARS = 1000

# Sanity bound on chunk_index read back from the evidence TSV. Real patients have
# single-digit chunk counts; anything beyond this is a misaligned row.
MAX_CHUNK_INDEX = 10_000

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "event_date",
    "date_source",
    "criterion_added",
    "criterion_label",
    "modality",
    "visceral_met_pattern",
    "cumulative_criteria",
    "num_criteria_to_date",
    "supporting_quote",
    "confidence",
    "source_note_date",
]

PROCESSED_COLUMNS = [
    "DFCI_MRN",
    "num_chunks",
    "num_chunks_ok",
    "num_criteria",
    "num_dropped",
    "status",
]

# Per-chunk log: the unit of resume. A chunk that failed is retried on the next
# run without re-calling the chunks that already succeeded.
CHUNK_COLUMNS = ["DFCI_MRN", "chunk_index", "num_criteria", "status"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the LLM on AVPC/NEPC evidence chunks and write a criteria timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing avpc_nepc_evidence.tsv and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help="Override path to avpc_nepc_evidence.tsv.")
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
        mrn = _to_numeric_scalar(row.get("DFCI_MRN"))
        idx = _to_numeric_scalar(row.get("chunk_index"))
        if mrn is None or idx is None:
            continue
        done.add((int(mrn), int(idx)))
    return done


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

    Returns (findings, error) where findings is a list of (finding_dict, vmp)
    tuples. Exactly one of the two is meaningful: on error findings is None.
    """
    payload = {
        "patient_mrn": int(mrn),
        "notes": [
            {"note_date": r["note_date"], "note_type": r["note_type"], "note_text": r["snippet"]}
            for r in chunk
        ],
    }
    messages = [
        {"role": "system", "content": NEPC_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
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
    found = result.get("criteria_found")
    if not isinstance(found, list):
        return None, "missing_criteria_found"
    vmp = result.get("visceral_met_pattern")
    vmp = vmp if vmp in VISCERAL_PATTERNS else "none"
    return [(f, vmp) for f in found if isinstance(f, dict)], None


def extract_patient(provider, client, model, max_retries, mrn, indexed_chunks):
    """Run one LLM call per chunk, keeping the findings from every chunk that works.

    `indexed_chunks` is a list of (chunk_index, chunk) pairs — only the chunks
    still outstanding for this patient, so a resumed run never re-calls a chunk
    that already succeeded.

    Returns (findings, chunk_results):
      findings      [(finding_dict, vmp, chunk_index), ...] for every chunk that
                    succeeded. A failing chunk never discards its siblings' work.
      chunk_results [{"chunk_index", "num_criteria", "status"}, ...], one per
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
                {"chunk_index": chunk_index, "num_criteria": 0, "status": error}
            )
            continue
        findings.extend((f, vmp, chunk_index) for f, vmp in chunk_findings)
        chunk_results.append({
            "chunk_index": chunk_index,
            "num_criteria": len(chunk_findings),
            "status": "ok",
        })
    return findings, chunk_results


def normalize_criterion(value):
    """Map a model-returned criterion onto its canonical label, or None.

    Recovers cheap near-misses (surrounding whitespace, lowercased "c2", a
    lowercased NEPC prefix) so they aren't counted as hallucinations. Anything
    that still doesn't match a known criterion returns None and is reported.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in VALID_CRITERIA:
        return text
    upper = text.upper()
    if upper in VALID_CRITERIA:
        return upper
    # "nepc:ne_features" -> "NEPC:ne_features" (prefix upper, suffix as-is)
    if ":" in text:
        prefix, _, suffix = text.partition(":")
        candidate = f"{prefix.strip().upper()}:{suffix.strip()}"
        if candidate in VALID_CRITERIA:
            return candidate
    return None


def raw_rows_from_findings(mrn, findings):
    """Build raw TSV rows from a patient's findings.

    Returns (rows, dropped) where `dropped` is a Counter of criterion values
    that could not be normalized onto a known criterion — these are silently
    discarded otherwise, which hides model hallucinations behind a lower
    num_criteria. Duplicate findings (same criterion + dates) are collapsed,
    since the prompt asks for each criterion ONCE but nothing enforces it.
    """
    rows = []
    dropped = Counter()
    seen = set()
    for finding, vmp, chunk_index in findings:
        criterion = normalize_criterion(finding.get("criterion"))
        if criterion is None:
            dropped[str(finding.get("criterion"))] += 1
            continue
        diagnosis_date = finding.get("diagnosis_date")
        source_note_date = finding.get("source_note_date")
        key = (criterion, diagnosis_date, source_note_date)
        if key in seen:
            continue
        seen.add(key)
        quote = flatten_ws(finding.get("quote"))
        if quote is not None and len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS].rstrip() + "..."
        rows.append({
            "DFCI_MRN": int(mrn),
            "chunk_index": chunk_index,
            "source_note_date": source_note_date,
            "criterion": criterion,
            "diagnosis_date": diagnosis_date,
            "modality": finding.get("modality"),
            "visceral_met_pattern": vmp if criterion == "C2" else None,
            "quote": quote,
            "confidence": finding.get("confidence"),
        })
    return rows, dropped


def _to_numeric_scalar(value):
    """Best-effort scalar -> float, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_timeline(raw_path, timeline_path):
    """Aggregate raw per-finding criteria into a per-patient onset timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pl.DataFrame(schema={c: pl.Utf8 for c in TIMELINE_COLUMNS}).write_csv(
            timeline_path, separator="\t"
        )
        return 0

    # Read every field as text and validate per row, so a single malformed/misaligned
    # row (e.g. free-text that shifted columns) can't abort the whole timeline build.
    raw = pl.read_csv(raw_path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)

    # For each (patient, criterion) keep the earliest documented occurrence as its onset.
    onsets = {}  # (mrn, criterion) -> establishing record
    skipped = 0
    for r in raw.iter_rows(named=True):
        criterion = r.get("criterion")
        if criterion not in VALID_CRITERIA:
            continue
        mrn_val = _to_numeric_scalar(r.get("DFCI_MRN"))
        if mrn_val is None:
            skipped += 1
            continue
        mrn = int(mrn_val)
        event_date, date_source = resolve_date(
            r.get("diagnosis_date"), r.get("source_note_date")
        )
        record = {
            "DFCI_MRN": mrn,
            "criterion_added": criterion,
            "criterion_label": CRITERION_LABELS[criterion],
            "event_date": event_date,
            "date_source": date_source,
            "modality": r.get("modality"),
            "visceral_met_pattern": r.get("visceral_met_pattern"),
            "supporting_quote": r.get("quote"),
            "confidence": r.get("confidence"),
            "source_note_date": r.get("source_note_date"),
        }
        key = (mrn, criterion)
        existing = onsets.get(key)
        # None dates sort last so any dated occurrence is preferred as the onset.
        if existing is None or (record["event_date"] or "9999-99-99") < (
            existing["event_date"] or "9999-99-99"
        ):
            onsets[key] = record

    if skipped:
        print(f"  Skipped {skipped} malformed/misaligned raw rows during timeline build")

    # Emit one row per onset, in chronological order per patient, with a cumulative set.
    rows = []
    by_patient = {}
    for record in onsets.values():
        by_patient.setdefault(record["DFCI_MRN"], []).append(record)

    for mrn in sorted(by_patient):
        events = sorted(
            by_patient[mrn],
            key=lambda rec: (rec["event_date"] or "9999-99-99", rec["criterion_added"]),
        )
        cumulative = []
        for rec in events:
            cumulative.append(rec["criterion_added"])
            out = dict(rec)
            out["cumulative_criteria"] = " | ".join(sorted(cumulative))
            out["num_criteria_to_date"] = len(cumulative)
            rows.append(out)

    if rows:
        timeline = pl.DataFrame({c: [row.get(c) for row in rows] for c in TIMELINE_COLUMNS})
    else:
        timeline = pl.DataFrame(schema={c: pl.Utf8 for c in TIMELINE_COLUMNS})
    timeline.write_csv(timeline_path, separator="\t")
    return timeline.height


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "avpc_nepc_evidence.tsv")
    raw_path = args.output_dir / "avpc_nepc_extractions_raw.tsv"
    chunk_log_path = args.output_dir / "avpc_nepc_processed_chunks.tsv"
    processed_path = args.output_dir / "avpc_nepc_processed_patients.tsv"
    timeline_path = args.output_dir / "avpc_nepc_timeline.tsv"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_nepc_notes.py first."
        )

    if args.overwrite:
        for path in (raw_path, chunk_log_path, processed_path, timeline_path):
            path.unlink(missing_ok=True)

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

    # Chunks are packed contiguously by collect_nepc_notes.py, but a filtered or
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
        dropped_total = Counter()

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
                rows, dropped = raw_rows_from_findings(mrn, findings)
                dropped_total.update(dropped)
                append_rows(raw_path, rows, RAW_COLUMNS)
                append_rows(
                    chunk_log_path,
                    [{"DFCI_MRN": int(mrn), **r} for r in chunk_results],
                    CHUNK_COLUMNS,
                )
                n_total = len(patient_chunks[mrn])
                n_ok = sum(1 for r in chunk_results if r["status"] == "ok")
                # Add back chunks skipped by resume (never in chunk_results this run).
                # Valid ONLY because read_done_chunks filters to status == "ok", so
                # "unattempted" and "already ok" are the same set today. If a future
                # skip reason is added (e.g. 1e's evidence-hash mismatch invalidating
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
                        "num_criteria": len(rows),
                        "num_dropped": int(sum(dropped.values())),
                        "status": status,
                    }],
                    PROCESSED_COLUMNS,
                )

        if dropped_total:
            total = sum(dropped_total.values())
            print(
                f"Dropped {total} findings with invalid criterion labels: "
                f"{dict(dropped_total.most_common(10))}"
            )

    # Compact once at the end rather than per patient: the hot loop stays
    # append-only (crash-safe), and retried patients leave one row, not one per run.
    compact_log(processed_path, PROCESSED_COLUMNS, ["DFCI_MRN"])
    compact_log(chunk_log_path, CHUNK_COLUMNS, ["DFCI_MRN", "chunk_index"])
    # Raw findings need a different collapse than the two logs above: a retried
    # chunk re-appends its findings, and without this they'd double-count in the
    # Phase 3 precision-sampling substrate (build_timeline itself is unaffected,
    # since it already collapses to earliest-per-(mrn, criterion)).
    dedupe_raw_findings(raw_path, RAW_COLUMNS, ["DFCI_MRN", "chunk_index"])

    n = build_timeline(raw_path, timeline_path)
    print(f"Wrote AVPC/NEPC criteria timeline ({n} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
