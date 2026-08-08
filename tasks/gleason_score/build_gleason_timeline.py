"""Stage 2 — Call the LLM on per-patient Gleason evidence chunks; write the timeline.

Reads gleason_evidence.parquet produced by collect_gleason_notes.py. Calls the LLM
once per chunk, and aggregates the findings into a deduped per-patient timeline.

Outputs (under <output-dir>):
  gleason_extractions_raw.parquet     per-finding extractions (provenance, pre-dedup)
  gleason_processed_chunks.parquet    per-chunk log — the unit of resume. Each row
                                   also records the evidence scan_config hash
                                   (read from gleason_evidence.meta.parquet) it was
                                   produced under, so a regenerated evidence file
                                   with different scan params can't silently
                                   "resume" onto now-mismatched chunk indices.
  gleason_processed_patients.parquet  processed-patient log (derived per-patient status)
  gleason_timeline.parquet            deduped timeline (every score + date per patient)

Resume guard: the evidence content, scan configuration, provider, model, prompt,
and output schemas are fingerprinted. Any mismatch raises rather than mixing
incompatible chunk results; re-run with --overwrite to intentionally start a
new extraction generation.

Usage:
  # Run collection first:
  python preprocessing/cli/collect_gleason_notes.py --output-dir /path/to/output

  # Then run LLM extraction:
  python tasks/gleason_score/build_gleason_timeline.py --output-dir /path/to/output
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import (  # noqa: E402
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_DATA_PATH,
    GLEASON_EVIDENCE_SCHEMA_VERSION,
)
from preprocessing.grounding import find_quote_support  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    derive_grade_group,
    file_sha256,
    flatten_ws,
    parse_stated_date,
    read_scan_config_meta,
    resolve_date,
)
from preprocessing.notes import load_selected_mrns, to_iso_date  # noqa: E402
from preprocessing.parquet_io import (  # noqa: E402
    append_rows_atomic,
    read_metadata,
    write_metadata,
    write_rows_atomic,
)
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.gleason_score.prompts import GLEASON_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_gleason_timeline"
GLEASON_EXTRACTION_SCHEMA_VERSION = "gleason-grounded-parquet-v3"

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

# Sanity bound on chunk_index read back from the evidence Parquet. Real patients have
# single-digit chunk counts; anything beyond this is a misaligned row.
MAX_CHUNK_INDEX = 10_000

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "gleason_date",
    "date_source",
    "date_precision",
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
    "run_config",
]

# Per-chunk log: the unit of resume. A chunk that failed is retried on the next
# run without re-calling the chunks that already succeeded. scan_config records
# the evidence hash (from gleason_evidence.meta.parquet) each row's chunk_index
# was assigned under, so a regenerated evidence file with different scan params
# can be detected before "resuming" onto now-mismatched chunks.
CHUNK_COLUMNS = [
    "DFCI_MRN",
    "chunk_index",
    "num_findings",
    "status",
    "scan_config",
    "run_config",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the LLM on Gleason evidence chunks and write a Gleason timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing gleason_evidence.parquet and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help="Override path to gleason_evidence.parquet.")
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
    """Append a complete row batch to an atomic Parquet artifact."""
    append_rows_atomic(path, rows, columns)


def read_done_chunks(path, run_config=None):
    """Return {(mrn, chunk_index)} for every chunk logged as status == "ok"."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    log = pl.read_parquet(path)
    if "DFCI_MRN" not in log.columns or "chunk_index" not in log.columns:
        return set()
    done = set()
    for row in log.iter_rows(named=True):
        if row.get("status") != "ok":
            continue
        if run_config is not None and row.get("run_config") != run_config:
            continue
        mrn = _to_int(row.get("DFCI_MRN"))
        idx = _to_int(row.get("chunk_index"))
        if mrn is None or idx is None:
            continue
        done.add((mrn, idx))
    return done


def _meta_path_for_evidence(evidence_path):
    return evidence_path.with_name(f"{evidence_path.stem}.meta.parquet")


def _gleason_run_fingerprint(evidence_path, provider_name, model):
    """Bind resume state to evidence, backend, prompt, and output contracts."""
    hasher = hashlib.sha256()
    for value in (
        file_sha256(evidence_path),
        provider_name,
        model,
        GLEASON_SYSTEM_PROMPT,
        CLINICAL_SAFETY_CONTEXT,
        GLEASON_EXTRACTION_SCHEMA_VERSION,
        json.dumps(RAW_COLUMNS),
        json.dumps(TIMELINE_COLUMNS),
        json.dumps(CHUNK_COLUMNS),
        json.dumps(PROCESSED_COLUMNS),
    ):
        hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()[:20]


def _validate_gleason_run(path, run_config, has_outputs, overwrite):
    if overwrite or not has_outputs:
        write_metadata(path, {"run_config": run_config})
        return
    if not path.exists():
        raise ValueError(
            "Existing Gleason outputs predate complete extraction fingerprinting. "
            "Re-run with --overwrite."
        )
    try:
        recorded = (read_metadata(path) or {}).get("run_config")
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Invalid Gleason run metadata: {path}") from exc
    if recorded != run_config:
        raise ValueError(
            "Gleason provider, model, prompt, schema, or evidence changed. "
            "Re-run with --overwrite to avoid mixed outputs."
        )


def check_scan_config(chunk_log_path, meta_path, evidence_path=None):
    """Guard chunk-index resume against a regenerated evidence file.

    The metadata sidecar and its evidence SHA-256 are mandatory. If a chunk log
    exists, its scan_config must match exactly; legacy or unverifiable state is
    rejected so chunk indices can never be resumed against different evidence.
    """
    meta = read_scan_config_meta(meta_path)
    current_config = meta.get("scan_config") if meta else None

    if current_config is None:
        raise ValueError(
            f"Evidence metadata is missing or invalid: {meta_path}. "
            "Regenerate Gleason evidence with --overwrite."
        )
    if meta.get("evidence_schema_version") != GLEASON_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "Gleason evidence predates the current cohort/evidence contract. "
            "Regenerate evidence with --overwrite."
        )
    if evidence_path is not None:
        recorded_digest = meta.get("evidence_sha256") if meta else None
        actual_digest = file_sha256(evidence_path)
        if not recorded_digest or recorded_digest != actual_digest:
            raise ValueError(
                "Gleason evidence content does not match its metadata sidecar. "
                "Regenerate evidence with --overwrite."
            )

    if not chunk_log_path.exists() or chunk_log_path.stat().st_size == 0:
        return current_config

    log = pl.read_parquet(chunk_log_path)
    if "scan_config" not in log.columns:
        raise ValueError(
            f"{chunk_log_path} predates safe scan fingerprinting. "
            "Re-run with --overwrite."
        )

    recorded_configs = set(log["scan_config"].drop_nulls().cast(pl.Utf8).to_list())
    if not recorded_configs:
        raise ValueError(
            f"{chunk_log_path} has no usable scan fingerprint. Re-run with --overwrite."
        )

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
    log = pl.read_parquet(path)
    if not all(c in log.columns for c in key_columns):
        return
    before = log.height
    compacted = log.unique(subset=key_columns, keep="last", maintain_order=True)
    if compacted.height == before:
        return
    write_rows_atomic(path, compacted.to_dicts(), columns)
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
    log = pl.read_parquet(path)
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
    write_rows_atomic(path, deduped.to_dicts(), columns)
    print(f"  Deduped {path.name}: {before} -> {deduped.height} rows")


def validate_gleason_finding(finding, chunk):
    """Validate one grading event and ground its quote/date in supplied evidence."""
    if not isinstance(finding, dict):
        return None, "gleason_finding_not_object"

    parsed = {
        name: _to_int(finding.get(name))
        for name in ("primary", "secondary", "total", "grade_group")
    }
    for name in ("primary", "secondary"):
        if finding.get(name) is not None and parsed[name] is None:
            return None, f"invalid_{name}:{finding.get(name)}"
        if parsed[name] is not None and not 1 <= parsed[name] <= 5:
            return None, f"out_of_range_{name}:{parsed[name]}"
    if finding.get("total") is not None and parsed["total"] is None:
        return None, f"invalid_total:{finding.get('total')}"
    if parsed["total"] is not None and not 2 <= parsed["total"] <= 10:
        return None, f"out_of_range_total:{parsed['total']}"
    if finding.get("grade_group") is not None and parsed["grade_group"] is None:
        return None, f"invalid_grade_group:{finding.get('grade_group')}"
    if parsed["grade_group"] is not None and not 1 <= parsed["grade_group"] <= 5:
        return None, f"out_of_range_grade_group:{parsed['grade_group']}"

    if parsed["primary"] is not None and parsed["secondary"] is not None:
        derived_total = parsed["primary"] + parsed["secondary"]
        if parsed["total"] is not None and parsed["total"] != derived_total:
            return None, "gleason_total_does_not_match_patterns"
        parsed["total"] = derived_total
        derived_group = derive_grade_group(parsed["primary"], parsed["secondary"])
        if (
            parsed["grade_group"] is not None
            and derived_group is not None
            and parsed["grade_group"] != derived_group
        ):
            return None, "grade_group_does_not_match_patterns"
    elif (parsed["primary"] is None) != (parsed["secondary"] is None):
        return None, "incomplete_gleason_pattern_pair"

    if parsed["total"] is None and parsed["grade_group"] is None:
        return None, "missing_gleason_score_and_grade_group"

    specimen_type = finding.get("specimen_type")
    if specimen_type not in {"biopsy", "prostatectomy", "TURP", "metastasis", "unknown"}:
        return None, f"invalid_specimen_type:{specimen_type}"
    if not isinstance(finding.get("is_historical_reference"), bool):
        return None, "invalid_is_historical_reference"

    claimed_source_date = finding.get("source_note_date")
    source_note_date = to_iso_date(claimed_source_date)
    if claimed_source_date not in (None, "", "null", "None") and source_note_date is None:
        return None, f"invalid_source_note_date:{claimed_source_date}"
    quote = finding.get("quote")
    support = find_quote_support(quote, chunk, claimed_date=source_note_date)
    if support is None or support.get("note_date") != source_note_date:
        return None, "quote_or_source_date_not_in_evidence"

    scoring_date = finding.get("scoring_date")
    if scoring_date not in (None, "", "null", "None"):
        stated_iso, _ = parse_stated_date(scoring_date)
        if stated_iso is None:
            return None, f"invalid_scoring_date:{scoring_date}"
        if source_note_date and stated_iso > source_note_date:
            return None, "scoring_date_after_source_note"

    normalized = dict(finding)
    normalized.update(parsed)
    normalized["source_note_date"] = source_note_date
    normalized["quote"] = flatten_ws(quote)
    return normalized, None


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
    normalized = []
    for finding in found:
        validated, validation_error = validate_gleason_finding(finding, chunk)
        if validation_error:
            return None, validation_error
        normalized.append(validated)
    return normalized, None


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
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def build_timeline(raw_path, timeline_path):
    """Resolve dates, validate, and de-duplicate raw extractions into the timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        write_rows_atomic(timeline_path, [], TIMELINE_COLUMNS)
        return 0

    # Read every field as text and validate per row, so a single malformed/misaligned
    # row (e.g. free-text that shifted columns) can't abort the whole timeline build.
    raw = pl.read_parquet(raw_path)
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
        grade_group = _to_int(r.get("grade_group"))
        if grade_group is not None and not (1 <= grade_group <= 5):
            grade_group = None

        if primary is not None and not (1 <= primary <= 5):
            invalid_score += 1
            continue
        if secondary is not None and not (1 <= secondary <= 5):
            invalid_score += 1
            continue
        # Gleason total is primary + secondary by definition; recompute it when
        # both patterns are present so an LLM arithmetic slip can't propagate.
        if primary is not None and secondary is not None:
            total = primary + secondary
        if total is not None and not (2 <= total <= 10):
            invalid_score += 1
            continue
        # Grade Group by itself is a valid prostate-grading event. Reject only
        # findings that provide neither a usable Gleason total nor Grade Group.
        if total is None and grade_group is None:
            invalid_score += 1
            continue

        if grade_group is None:
            grade_group = derive_grade_group(primary, secondary)

        gleason_date, date_source, date_precision = resolve_date(
            r.get("scoring_date"), r.get("source_note_date")
        )
        specimen_type = r.get("specimen_type")

        key = (
            mrn,
            primary,
            secondary,
            total,
            grade_group,
            gleason_date,
            specimen_type,
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "DFCI_MRN": mrn,
            "gleason_date": gleason_date,
            "date_source": date_source,
            "date_precision": date_precision,
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
    write_rows_atomic(timeline_path, timeline.to_dicts(), TIMELINE_COLUMNS)
    return timeline.height


def _load_patient_chunks(evidence_df):
    """Load evidence without changing its persisted chunk identifiers."""
    required = {"DFCI_MRN", "chunk_index", "note_date", "note_type", "snippet"}
    missing = sorted(required - set(evidence_df.columns))
    if missing:
        raise ValueError(f"Evidence table missing required columns: {missing}")

    patient_chunks = {}
    invalid = 0
    for row in evidence_df.iter_rows(named=True):
        mrn = _to_int(row.get("DFCI_MRN"))
        chunk_index = _to_int(row.get("chunk_index"))
        snippet = row.get("snippet") or ""
        if (
            mrn is None
            or chunk_index is None
            or not 0 <= chunk_index <= MAX_CHUNK_INDEX
            or not str(snippet).strip()
        ):
            invalid += 1
            continue
        patient_chunks.setdefault(mrn, {}).setdefault(chunk_index, []).append(
            {
                "note_date": row.get("note_date"),
                "note_type": row.get("note_type") or "Unknown",
                "snippet": snippet,
            }
        )
    if invalid:
        print(f"  Skipped {invalid} invalid evidence rows")
    return patient_chunks


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "gleason_evidence.parquet")
    meta_path = _meta_path_for_evidence(evidence_path)
    raw_path = args.output_dir / "gleason_extractions_raw.parquet"
    chunk_log_path = args.output_dir / "gleason_processed_chunks.parquet"
    processed_path = args.output_dir / "gleason_processed_patients.parquet"
    timeline_path = args.output_dir / "gleason_timeline.parquet"
    run_meta_path = args.output_dir / "gleason_run.parquet"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_gleason_notes.py first."
        )

    if args.overwrite:
        for path in (
            raw_path,
            chunk_log_path,
            processed_path,
            timeline_path,
            run_meta_path,
        ):
            path.unlink(missing_ok=True)

    provider = get_provider(args.provider)
    model = args.model or provider.default_model
    run_config = _gleason_run_fingerprint(
        evidence_path, args.provider, model
    )
    _validate_gleason_run(
        run_meta_path,
        run_config,
        any(
            path.exists()
            for path in (raw_path, chunk_log_path, processed_path, timeline_path)
        ),
        args.overwrite,
    )

    # Must run before read_done_chunks: unverifiable or changed evidence cannot
    # reuse persisted chunk indices.
    scan_config = check_scan_config(
        chunk_log_path, meta_path, evidence_path=evidence_path
    )
    print(f"Extraction fingerprint: {run_config} ({args.provider}/{model})")

    evidence_df = pl.read_parquet(evidence_path)
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

    patient_chunks = _load_patient_chunks(evidence_df)

    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients to process: {len(patient_chunks)} "
        f"({total_chunks} LLM calls across chunks)"
    )

    # Resume at chunk granularity: a patient whose chunk 2 failed re-runs only
    # chunk 2, keeping the findings chunks 0 and 1 already produced.
    done_chunks = read_done_chunks(chunk_log_path, run_config)
    if done_chunks:
        print(f"Already completed chunks: {len(done_chunks)}")

    todo = []
    for mrn in sorted(patient_chunks):
        outstanding = [
            (chunk_index, chunk)
            for chunk_index, chunk in sorted(patient_chunks[mrn].items())
            if (mrn, chunk_index) not in done_chunks
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
                    [
                        {
                            "DFCI_MRN": int(mrn),
                            "scan_config": scan_config,
                            "run_config": run_config,
                            **r,
                        }
                        for r in chunk_results
                    ],
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
                        "run_config": run_config,
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
