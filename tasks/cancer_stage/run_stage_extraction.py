"""Stage 2 — Call the LLM on per-patient evidence chunks; write the stage timeline.

Reads stage_evidence.parquet produced by extract_stage_notes.py. Groups snippets by
patient into payload-sized chunks (chronological, greedy packing), calls the LLM
once per chunk, and aggregates the findings into a deduped stage timeline.

Outputs (under <output-dir>):
  stage_extractions_raw.parquet  Per-finding extractions (one row per LLM finding,
                                 pre-dedup, with rationale for auditing).
  stage_processed_patients.parquet Per-patient processing log (resumability + failures).
  stage_timeline.parquet         Deduped stage timeline — one row per distinct staging
                                 event per patient.

Usage:
  # Run scan first:
  python preprocessing/cli/extract_stage_notes.py --output-dir /path/to/output

  # Then run LLM extraction:
  python tasks/cancer_stage/run_stage_extraction.py --output-dir /path/to/output
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import CLINICAL_SAFETY_CONTEXT, SNIPPET_PROFILES  # noqa: E402
from preprocessing.grounding import find_quote_support  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    file_sha256,
    flatten_ws,
    parse_stated_date,
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
from tasks.cancer_stage.prompts import STAGE_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("STAGE_OUTPUT_DIR", "/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/")
)
DEFAULT_PAYLOAD_MAX_CHARS = SNIPPET_PROFILES["longitudinal"].payload_max_chars

RAW_COLUMNS = [
    "DFCI_MRN",
    "source_note_date",
    "cancer_type",
    "staging_system",
    "stage_raw",
    "stage_group",
    "stage_date",
    "is_historical_reference",
    "supporting_quote",
    "confidence",
    "rationale",
]

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "cancer_type",
    "staging_system",
    "stage_raw",
    "stage_group",
    "stage_date",
    "date_source",
    "date_precision",
    "is_historical_reference",
    "supporting_quote",
    "confidence",
    "source_note_date",
]

PROCESSED_COLUMNS = ["DFCI_MRN", "num_chunks", "num_findings", "status"]
STAGE_EXTRACTION_SCHEMA_VERSION = "cancer-stage-grounded-parquet-v3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the LLM on stage evidence chunks and write a stage timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing stage_evidence.parquet and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help="Override path to stage_evidence.parquet.")
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="Process only these MRNs (Parquet with a DFCI_MRN column).")
    parser.add_argument("--mrns", default=None,
                        help="Comma- or space-separated MRNs to process.")
    parser.add_argument("--payload-max-chars", type=int, default=DEFAULT_PAYLOAD_MAX_CHARS,
                        help="Max snippet chars packed into one LLM call (one chunk per patient "
                             "until the budget is full).")
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
    parser.add_argument("--limit-patients", type=int, default=None,
                        help="Process at most this many patients (useful for pilots).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing raw/processed/timeline files before running.")
    return parser.parse_args()


def group_evidence_chunks(evidence_df, payload_max_chars):
    """Group evidence rows by patient into chronological, payload-sized chunks.

    Returns {mrn: [chunk, ...]}, where each chunk is a list of snippet dicts.
    No snippet is ever dropped — a patient with very long evidence gets multiple
    chunks and therefore multiple LLM calls.
    """
    by_mrn = {}
    for row in evidence_df.iter_rows(named=True):
        mrn = int(row["DFCI_MRN"])
        trigger_categories = row.get("trigger_categories")
        by_mrn.setdefault(mrn, []).append({
            "note_date": row.get("note_date"),
            "note_type": row.get("note_type") or "Unknown",
            "trigger_categories": (
                list(trigger_categories)
                if isinstance(trigger_categories, (list, tuple))
                else str(trigger_categories).split(",")
                if trigger_categories
                else []
            ),
            "snippet": row.get("snippet") or "",
        })

    patient_chunks = {}
    for mrn, recs in by_mrn.items():
        recs.sort(key=lambda r: (r["note_date"] or "9999-99-99"))
        chunks, current, current_len = [], [], 0
        for rec in recs:
            slen = len(rec["snippet"])
            if current and current_len + slen > payload_max_chars:
                chunks.append(current)
                current, current_len = [], 0
            current.append(rec)
            current_len += slen
        if current:
            chunks.append(current)
        patient_chunks[mrn] = chunks
    return patient_chunks


def extract_patient(provider, client, model, max_retries, mrn, chunks):
    """Run one LLM call per chunk; return the merged findings list for the patient."""
    findings = []
    for chunk in chunks:
        payload = {
            "patient_mrn": int(mrn),
            "stage_contexts": [
                {
                    "note_date": r["note_date"],
                    "note_type": r["note_type"],
                    "trigger_categories": r["trigger_categories"],
                    "note_text": r["snippet"],
                }
                for r in chunk
            ],
        }
        messages = [
            {"role": "system", "content": STAGE_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response_text, error = provider.call_with_retry(client, model, messages, max_retries)
        if error:
            # Never mark a patient complete when any evidence chunk was rejected.
            # Doing so would silently turn a partial extraction into a permanent
            # success that resumability would skip on every subsequent run.
            return None, error
        try:
            result = parse_json_response(response_text)
        except json.JSONDecodeError as exc:
            return None, f"json_parse: {exc}"
        if not isinstance(result, dict):
            return None, f"non_dict_response: {type(result).__name__}"
        chunk_findings = result.get("stage_findings")
        if not isinstance(chunk_findings, list):
            return None, "missing_stage_findings"
        for finding in chunk_findings:
            normalized, validation_error = validate_stage_finding(finding, chunk)
            if validation_error:
                return None, validation_error
            findings.append(normalized)
    return findings, None


_VALID_STAGES = {"I", "II", "III", "IV"}
_WORD_TO_STAGE = {"ONE": "I", "TWO": "II", "THREE": "III", "FOUR": "IV"}


def _normalize_stage_group(val):
    """Normalize base/substage Roman, Arabic, or word values to I/II/III/IV."""
    if not val:
        return None
    cleaned = re.sub(r"(?i)^stage\s+", "", str(val).strip()).upper()
    match = re.match(r"^(IV|III|II|I|[1-4]|ONE|TWO|THREE|FOUR)", cleaned)
    if not match:
        return None
    cleaned = match.group(1)
    cleaned = _WORD_TO_STAGE.get(cleaned, cleaned)
    cleaned = {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(cleaned, cleaned)
    return cleaned if cleaned in _VALID_STAGES else None


def validate_stage_finding(finding, chunk):
    """Validate one stage event and ground its quote/date in the current chunk."""
    if not isinstance(finding, dict):
        return None, "stage_finding_not_object"
    cancer_type = finding.get("cancer_type")
    if not isinstance(cancer_type, str) or not cancer_type.strip():
        return None, "missing_cancer_type"
    stage_raw = finding.get("stage_raw") or finding.get("stage_group")
    if not isinstance(stage_raw, str) or not stage_raw.strip():
        return None, "missing_stage_raw"
    staging_system = finding.get("staging_system")
    if staging_system is not None and not isinstance(staging_system, str):
        return None, "invalid_staging_system"

    supplied_group = finding.get("stage_group")
    normalized_group = _normalize_stage_group(supplied_group) if supplied_group else None
    if supplied_group not in (None, "") and normalized_group is None:
        return None, f"invalid_stage_group:{supplied_group}"
    raw_group = _normalize_stage_group(stage_raw)
    if raw_group and normalized_group and raw_group != normalized_group:
        return None, "stage_group_does_not_match_stage_raw"

    claimed_source_date = finding.get("source_note_date")
    source_note_date = to_iso_date(claimed_source_date)
    if claimed_source_date not in (None, "", "null", "None") and source_note_date is None:
        return None, f"invalid_source_note_date:{claimed_source_date}"
    quote = finding.get("supporting_quote")
    support = find_quote_support(quote, chunk, claimed_date=source_note_date)
    if support is None or support.get("note_date") != source_note_date:
        return None, "supporting_quote_or_source_date_not_in_evidence"

    stage_date = finding.get("stage_date")
    if stage_date not in (None, "", "null", "None"):
        stated_iso, _ = parse_stated_date(stage_date)
        if stated_iso is None:
            return None, f"invalid_stage_date:{stage_date}"
        if source_note_date and stated_iso > source_note_date:
            return None, "stage_date_after_source_note"

    if not isinstance(finding.get("is_historical_reference"), bool):
        return None, "invalid_is_historical_reference"
    if finding.get("confidence") not in {"high", "medium", "low"}:
        return None, f"invalid_confidence:{finding.get('confidence')}"
    rationale = finding.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None, "missing_rationale"

    normalized = dict(finding)
    normalized.update(
        {
            "cancer_type": cancer_type.strip(),
            "staging_system": staging_system.strip() if staging_system else None,
            "stage_raw": stage_raw.strip(),
            "stage_group": normalized_group,
            "source_note_date": source_note_date,
            "supporting_quote": flatten_ws(quote),
        }
    )
    return normalized, None


def raw_rows_from_findings(mrn, findings):
    rows = []
    for finding in findings:
        rows.append({
            "DFCI_MRN": int(mrn),
            "source_note_date": finding.get("source_note_date"),
            "cancer_type": finding.get("cancer_type"),
            "staging_system": finding.get("staging_system"),
            "stage_raw": finding.get("stage_raw") or finding.get("stage_group"),
            "stage_group": _normalize_stage_group(
                finding.get("stage_group") or finding.get("stage_raw")
            ),
            "stage_date": finding.get("stage_date"),
            "is_historical_reference": finding.get("is_historical_reference"),
            "supporting_quote": flatten_ws(finding.get("supporting_quote")),
            "confidence": finding.get("confidence"),
            "rationale": flatten_ws(finding.get("rationale")),
        })
    return rows


def append_rows(path, rows, columns):
    """Append a complete row batch to an atomic Parquet artifact."""
    append_rows_atomic(path, rows, columns)


def _str(val):
    """Return val as a stripped string, treating None and NaN as empty string."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


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
    """Deduplicate raw findings into the stage timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        write_rows_atomic(timeline_path, [], TIMELINE_COLUMNS)
        return 0

    raw = pl.read_parquet(raw_path)
    seen = set()
    rows = []
    for r in raw.iter_rows(named=True):
        mrn_val = _to_numeric_scalar(r.get("DFCI_MRN"))
        if mrn_val is None:
            continue
        mrn = int(mrn_val)

        # Normalize dedup key fields so formatting differences don't create duplicates.
        cancer_type_raw = _str(r.get("cancer_type"))
        staging_system_raw = _str(r.get("staging_system"))
        stage_raw = _str(r.get("stage_raw"))
        stage_group_raw = _str(r.get("stage_group"))

        stage_date, date_source, date_precision = resolve_date(
            r.get("stage_date"), r.get("source_note_date")
        )

        key = (
            mrn,
            cancer_type_raw.lower() or None,
            staging_system_raw.lower() or None,
            stage_raw.lower() or None,
            stage_group_raw.upper() or None,
            stage_date,
        )
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "DFCI_MRN": mrn,
            "cancer_type": cancer_type_raw or None,
            "staging_system": staging_system_raw or None,
            "stage_raw": stage_raw or None,
            "stage_group": stage_group_raw or None,
            "stage_date": stage_date,
            "date_source": date_source,
            "date_precision": date_precision,
            "is_historical_reference": r.get("is_historical_reference"),
            "supporting_quote": r.get("supporting_quote"),
            "confidence": r.get("confidence"),
            "source_note_date": r.get("source_note_date"),
        })

    if not rows:
        timeline = pl.DataFrame(schema={c: pl.Utf8 for c in TIMELINE_COLUMNS})
    else:
        timeline = pl.DataFrame({c: [row.get(c) for row in rows] for c in TIMELINE_COLUMNS})
        timeline = timeline.sort(
            ["DFCI_MRN", "cancer_type", "staging_system", "stage_date"], nulls_last=True
        )
        # Keep only rows where stage_group changes within each (patient, cancer_type).
        # This collapses repeated identical staging entries over time — once a stage
        # is established (including metastatic/IV), subsequent rows with the same
        # stage add no new information.
        last_stage = {}
        keep_mask = []
        for row in timeline.iter_rows(named=True):
            key = (
                row["DFCI_MRN"],
                (_str(row["cancer_type"])).lower(),
                (_str(row["staging_system"])).lower(),
            )
            curr = (
                (_str(row["stage_group"])).upper(),
                (_str(row["stage_raw"])).lower(),
            )
            if last_stage.get(key) != curr:
                keep_mask.append(True)
                last_stage[key] = curr
            else:
                keep_mask.append(False)
        timeline = timeline.filter(pl.Series(keep_mask))
    write_rows_atomic(timeline_path, timeline.to_dicts(), TIMELINE_COLUMNS)
    return timeline.height


def _stage_run_fingerprint(evidence_path, provider_name, model, payload_max_chars):
    hasher = hashlib.sha256()
    for value in (
        file_sha256(evidence_path),
        provider_name,
        model,
        STAGE_SYSTEM_PROMPT,
        CLINICAL_SAFETY_CONTEXT,
        STAGE_EXTRACTION_SCHEMA_VERSION,
        json.dumps(RAW_COLUMNS),
        json.dumps(TIMELINE_COLUMNS),
        str(int(payload_max_chars)),
    ):
        hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()[:20]


def _validate_stage_run(path, run_config, has_outputs, overwrite):
    if overwrite or not has_outputs:
        write_metadata(path, {"run_config": run_config})
        return
    if not path.exists():
        raise ValueError(
            "Existing staging outputs predate run fingerprinting. Re-run with --overwrite."
        )
    try:
        recorded = (read_metadata(path) or {}).get("run_config")
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Invalid staging run metadata: {path}") from exc
    if recorded != run_config:
        raise ValueError(
            f"Staging evidence/config changed ({recorded} != {run_config}). "
            "Re-run with --overwrite."
        )


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "stage_evidence.parquet")
    raw_path = args.output_dir / "stage_extractions_raw.parquet"
    processed_path = args.output_dir / "stage_processed_patients.parquet"
    timeline_path = args.output_dir / "stage_timeline.parquet"
    run_meta_path = args.output_dir / "stage_run.parquet"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/extract_stage_notes.py first."
        )

    if args.overwrite:
        for path in (raw_path, processed_path, timeline_path, run_meta_path):
            path.unlink(missing_ok=True)

    provider = get_provider(args.provider)
    model = args.model or provider.default_model
    run_config = _stage_run_fingerprint(
        evidence_path, args.provider, model, args.payload_max_chars
    )
    _validate_stage_run(
        run_meta_path,
        run_config,
        any(path.exists() for path in (raw_path, processed_path, timeline_path)),
        args.overwrite,
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

    patient_chunks = group_evidence_chunks(evidence_df, args.payload_max_chars)
    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients to process: {len(patient_chunks)} "
        f"({total_chunks} LLM calls across chunks)"
    )

    completed = set()
    if processed_path.exists() and processed_path.stat().st_size > 0:
        log = pl.read_parquet(processed_path)
        completed = set(
            log.filter(pl.col("status") == "ok")["DFCI_MRN"].cast(pl.Int64).to_list()
        )
    print(f"Already completed patients: {len(completed)}")

    todo = [m for m in sorted(patient_chunks) if m not in completed]
    if args.limit_patients is not None:
        todo = todo[: args.limit_patients]
    print(f"Patients to extract with LLM: {len(todo)}")

    if todo:
        client = provider.build_client()

        def worker(mrn):
            chunks = patient_chunks[mrn]
            findings, error = extract_patient(
                provider, client, model, args.max_retries, mrn, chunks
            )
            return mrn, len(chunks), findings, error

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(worker, m): m for m in todo}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Patients", unit="pt"
            ):
                mrn, n_chunks, findings, error = future.result()
                if error or findings is None:
                    append_rows(
                        processed_path,
                        [{"DFCI_MRN": int(mrn), "num_chunks": n_chunks,
                          "num_findings": 0, "status": error or "no_result"}],
                        PROCESSED_COLUMNS,
                    )
                    continue
                rows = raw_rows_from_findings(mrn, findings)
                append_rows(raw_path, rows, RAW_COLUMNS)
                append_rows(
                    processed_path,
                    [{"DFCI_MRN": int(mrn), "num_chunks": n_chunks,
                      "num_findings": len(rows), "status": "ok"}],
                    PROCESSED_COLUMNS,
                )

    n = build_timeline(raw_path, timeline_path)
    print(f"Wrote stage timeline ({n} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
