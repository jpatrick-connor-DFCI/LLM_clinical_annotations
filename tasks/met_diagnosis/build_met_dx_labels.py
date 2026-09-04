"""Adjudicate a single metastatic prostate cancer label + first-mention date per patient.

Stage 1 (`preprocessing/cli/collect_met_dx_notes.py`) writes trigger-bearing
evidence chunks. This stage maps every chunk into candidate metastatic-disease
statements, then adjudicates all candidates for a patient into one yes/no label
with the earliest documented metastasis date. Chunk maps and patient
adjudications are durable resume state.

Precision-over-recall properties, shared with tasks/nepc_diagnosis:
  - Quote grounding requires source-date EQUALITY (via
    preprocessing.grounding.find_quote_support). A quote grounded in a
    differently-dated note is rejected, never silently re-dated.
  - Every grounded quote must clear the deterministic gate in
    tasks/met_diagnosis/veto.py, which the model cannot override.
  - The adjudication stage may only cite a quote that came from a validated
    candidate; an invented quote downgrades the patient to negative.

What differs from the NEPC diagnosis task:
  - `met_site` is validated as an enum AND corroborated against the grounded
    quote, because the bone/visceral/distant-nodal vs. regional-nodal
    distinction is what separates M1 from N1 disease.
  - The note-date fallback is the COMMON path, not an edge case: metastatic
    disease is rarely given an explicit date in text, so most labels resolve
    through _earliest_qualifying_note_date. See its docstring.

The resume fingerprint includes the evidence scan, provider, model, prompt
texts, prompt schema version, output columns, and the veto version. Changing any
of them requires `--overwrite`; this prevents mixed-model, mixed-prompt, or
mixed-gate research outputs.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
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
    MET_DX_EVIDENCE_SCHEMA_VERSION,
)
from preprocessing.grounding import find_quote_support, quote_core  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    flatten_ws,
    file_sha256,
    parse_stated_date,
    read_scan_config_meta,
    resolve_date,
)
from preprocessing.notes import load_selected_mrns, to_iso_date  # noqa: E402
from preprocessing.parquet_io import (  # noqa: E402
    append_rows_atomic,
    write_rows_atomic,
)
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.met_diagnosis.prompts import (  # noqa: E402
    MET_DX_MAP_PROMPT,
    MET_DX_SYNTHESIS_PROMPT,
    PROMPT_SCHEMA_VERSION,
)
from tasks.met_diagnosis.veto import VETO_VERSION, screen_quote  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_met_diagnosis"

VALID_EVIDENCE_TYPES = {
    "stated_metastatic_disease",
    "imaging_metastasis",
    "pathologic_metastasis",
    "m_stage",
}
VALID_ASSERTION_TYPES = {"established", "reported_history"}
VALID_MODALITIES = {"imaging", "pathology", "clinical"}
VALID_MET_SITES = {"bone", "visceral", "distant_nodal", "unspecified"}
VALID_CONFIDENCE = {"high", "medium", "low"}
SUCCESS_STATUSES = {"ok", "ok_with_rejections"}
MAX_QUOTE_CHARS = 1000
MAX_CHUNK_INDEX = 10_000
MAX_CANDIDATES_PER_CHUNK = 20

# Corroboration patterns for met_site. A claimed site must be visible in the
# quote the model actually grounded, in the spirit of cancer_stage's
# _value_is_grounded -- the model may not assert a site the text does not name.
# "unspecified" is exempt: it asserts the absence of a named site.
MET_SITE_PATTERNS = {
    "bone": re.compile(
        r"\b(?:bone|osseous|skeletal|spine|spinal|vertebr\w*|rib|femur|femoral|"
        r"pelvis|pelvic|sacrum|sacral|ilium|iliac|humerus|sternum|scapula|"
        r"skull|calvari\w*|acetabul\w*|sclerotic|lytic|m1b)\b",
        flags=re.IGNORECASE,
    ),
    "visceral": re.compile(
        r"\b(?:liver|hepatic|lung|pulmonary|adrenal|brain|cerebral|"
        r"pleural|peritoneal|visceral|m1c)\b",
        flags=re.IGNORECASE,
    ),
    "distant_nodal": re.compile(
        r"\b(?:lymph\s*node[s]?|nodal|node[s]?|lymphadenopathy|adenopathy|"
        r"retroperitoneal|para[\s-]?aortic|paraaortic|mediastinal|"
        r"supraclavicular|inguinal|axillary|m1a)\b",
        flags=re.IGNORECASE,
    ),
}

CANDIDATE_COLUMNS = [
    "DFCI_MRN",
    "chunk_index",
    "source_note_date",
    "evidence_type",
    "assertion_type",
    "met_site",
    "stated_metastasis_date",
    "modality",
    "quote",
    "confidence",
]

LABEL_COLUMNS = [
    "DFCI_MRN",
    "has_metastatic_disease",
    "first_metastasis_date",
    "date_source",
    "date_precision",
    "stated_metastasis_date",
    "source_note_date",
    "evidence_type",
    "assertion_type",
    "met_site",
    "modality",
    "supporting_quote",
    "confidence",
    "rationale",
    "num_candidates",
    "label_source",
]

CHUNK_COLUMNS = [
    "DFCI_MRN",
    "chunk_index",
    "num_candidates",
    "num_dropped",
    "status",
    "scan_config",
    "run_config",
    "result_json",
]

PROCESSED_COLUMNS = [
    "DFCI_MRN",
    "num_chunks",
    "num_chunks_ok",
    "num_candidates",
    "has_metastatic_disease",
    "num_dropped",
    "status",
    "run_config",
    "result_json",
]

REJECTED_COLUMNS = [
    "DFCI_MRN",
    "stage",
    "chunk_index",
    "reason",
    "item_json",
    "run_config",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adjudicate one metastatic prostate cancer label and date per patient."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-path", type=Path, default=None)
    parser.add_argument(
        "--evidence-meta-path",
        type=Path,
        default=None,
        help="Override evidence metadata path. By default it is derived from --evidence-path.",
    )
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="CSV cohort file containing DFCI_MRN values.")
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--provider", choices=["dfci_gpt", "vertex_ai"], default="dfci_gpt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument(
        "--rebuild-labels-only",
        action="store_true",
        help=(
            "Rebuild candidate/audit/label Parquets only from saved successful "
            "adjudications; never make an LLM call or retry incomplete patients."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.limit_patients is not None and args.limit_patients < 0:
        parser.error("--limit-patients must be >= 0")
    return args


def append_rows(path, rows, columns):
    """Append complete Parquet records; callers compact them afterward."""
    append_rows_atomic(path, rows, columns)


def _write_rows_atomic(path, rows, columns):
    write_rows_atomic(path, rows, columns)


def compact_log(path, columns, key_columns):
    if not path.exists() or path.stat().st_size == 0:
        return
    log = pl.read_parquet(path)
    if not all(c in log.columns for c in key_columns):
        return
    compacted = log.unique(subset=key_columns, keep="last", maintain_order=True)
    _write_rows_atomic(path, compacted.to_dicts(), columns)


def _to_exact_int(value):
    """Parse an integer without Float64 precision loss or decimal truncation."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _meta_path_for_evidence(evidence_path):
    return evidence_path.with_name(f"{evidence_path.stem}.meta.parquet")


def _normalized_text(value):
    return flatten_ws(value) or ""


def extraction_run_config(scan_config, provider_name, model):
    """Fingerprint every input that can change a label's meaning.

    VETO_VERSION is included deliberately: the deterministic gate decides
    positives as much as the prompt does, so changing it must force --overwrite
    rather than mixing labels adjudicated under two different gates.
    """
    payload = {
        "schema": PROMPT_SCHEMA_VERSION,
        "scan_config": scan_config,
        "provider": provider_name,
        "model": model,
        "map_prompt": MET_DX_MAP_PROMPT,
        "synthesis_prompt": MET_DX_SYNTHESIS_PROMPT,
        "safety_context": CLINICAL_SAFETY_CONTEXT,
        "veto": VETO_VERSION,
        "columns": [
            CANDIDATE_COLUMNS,
            LABEL_COLUMNS,
            CHUNK_COLUMNS,
            PROCESSED_COLUMNS,
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def check_resume_config(
    chunk_log_path, meta_path, provider_name, model, evidence_path=None
):
    """Return `(scan_config, run_config)` or reject unsafe/legacy resume state."""
    meta = read_scan_config_meta(meta_path)
    scan_config = meta.get("scan_config") if isinstance(meta, dict) else None
    if not scan_config:
        raise ValueError(
            f"Evidence metadata is missing or invalid: {meta_path}. "
            "Regenerate evidence so its content/configuration can be verified."
        )
    if meta.get("evidence_schema_version") != MET_DX_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "Metastasis evidence predates the current trigger/cohort contract. "
            "Regenerate evidence with --overwrite."
        )
    if evidence_path is not None:
        recorded_digest = meta.get("evidence_sha256")
        actual_digest = file_sha256(evidence_path)
        if not recorded_digest or recorded_digest != actual_digest:
            raise ValueError(
                "Evidence content does not match its metadata sidecar. "
                "Regenerate evidence before extraction."
            )
    run_config = extraction_run_config(scan_config, provider_name, model)
    if not chunk_log_path.exists() or chunk_log_path.stat().st_size == 0:
        return scan_config, run_config

    log = pl.read_parquet(chunk_log_path)
    required = {"scan_config", "run_config"}
    if not required.issubset(log.columns):
        raise ValueError(
            f"{chunk_log_path} predates safe extraction fingerprints. "
            "Re-run with --overwrite."
        )
    recorded_scan = set(log["scan_config"].drop_nulls().cast(pl.Utf8).to_list())
    recorded_run = set(log["run_config"].drop_nulls().cast(pl.Utf8).to_list())
    if recorded_scan and recorded_scan != {scan_config}:
        raise ValueError(
            "Evidence differs from the existing chunk log. Re-run with --overwrite."
        )
    if recorded_run and recorded_run != {run_config}:
        raise ValueError(
            "Provider, model, prompt, schema, veto version, or evidence changed "
            "since the existing chunk log was created. Re-run with --overwrite "
            "to avoid mixed outputs."
        )
    return scan_config, run_config


def read_done_chunks(path, run_config=None):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    log = pl.read_parquet(path)
    required = {"DFCI_MRN", "chunk_index", "status"}
    if not required.issubset(log.columns):
        return set()
    done = set()
    for row in log.iter_rows(named=True):
        if row.get("status") not in SUCCESS_STATUSES:
            continue
        if run_config is not None and row.get("run_config") != run_config:
            continue
        mrn = _to_exact_int(row.get("DFCI_MRN"))
        idx = _to_exact_int(row.get("chunk_index"))
        if mrn is not None and idx is not None:
            done.add((mrn, idx))
    return done


def _normalize_enum(value, allowed):
    text = _normalized_text(value).lower().replace(" ", "_").replace("-", "_")
    return text if text in allowed else None


def _validate_grounded_quote(item, notes):
    """Ground a quote against `notes` with strict source-date equality.

    Returns (quote, source_note_date, error). A quote that occurs only in a
    differently-dated note is rejected outright rather than having its
    provenance silently corrected -- the date IS the deliverable here, so
    re-dating a finding would corrupt the answer rather than repair it.
    """
    quote = _normalized_text(item.get("quote"))
    if not quote:
        return None, None, "missing_quote"
    source_note_date = to_iso_date(item.get("source_note_date"))
    if item.get("source_note_date") not in (None, "", "null", "None") and (
        source_note_date is None
    ):
        return None, None, f"invalid_source_note_date:{item.get('source_note_date')}"
    support = find_quote_support(quote, notes, claimed_date=source_note_date)
    if support is None or support.get("note_date") != source_note_date:
        # find_quote_support requires >= 8 normalized chars; surface that cause
        # separately so it does not hide inside the generic grounding failure.
        if len(quote_core(quote)) < 8:
            return None, None, "quote_too_short"
        return None, None, "quote_or_source_date_not_in_evidence"
    return quote, source_note_date, None


def _validate_common_fields(item, notes):
    """Shared enum/grounding/site/date/veto gauntlet for map and synthesis findings."""
    if not isinstance(item, dict):
        return None, "finding_not_object"
    evidence_type = _normalize_enum(item.get("evidence_type"), VALID_EVIDENCE_TYPES)
    if evidence_type is None:
        return None, f"invalid_evidence_type:{item.get('evidence_type')}"
    assertion_type = _normalize_enum(item.get("assertion_type"), VALID_ASSERTION_TYPES)
    if assertion_type is None:
        return None, f"invalid_assertion_type:{item.get('assertion_type')}"
    met_site = _normalize_enum(item.get("met_site"), VALID_MET_SITES)
    if met_site is None:
        return None, f"invalid_met_site:{item.get('met_site')}"
    # No note-type fallback repair for modality: a strict task must not
    # synthesize metadata the model failed to supply.
    modality = _normalize_enum(item.get("modality"), VALID_MODALITIES)
    if modality is None:
        return None, f"invalid_modality:{item.get('modality')}"
    confidence = _normalize_enum(item.get("confidence"), VALID_CONFIDENCE)
    if confidence is None:
        return None, f"invalid_confidence:{item.get('confidence')}"

    quote, source_note_date, error = _validate_grounded_quote(item, notes)
    if error:
        return None, error

    # The claimed site must be visible in the grounded quote. This is what makes
    # the M1-vs-N1 boundary auditable: a "distant_nodal" claim on a quote that
    # names only pelvic nodes fails here even before the veto's
    # regional_nodal_only check.
    site_pattern = MET_SITE_PATTERNS.get(met_site)
    if site_pattern is not None and site_pattern.search(quote) is None:
        return None, "met_site_not_in_quote"

    stated_raw = item.get("stated_metastasis_date")
    stated_iso = None
    if stated_raw not in (None, "", "null", "None"):
        stated_iso, _ = parse_stated_date(stated_raw)
        if stated_iso is None:
            return None, f"invalid_stated_metastasis_date:{stated_raw}"
        if source_note_date and stated_iso > source_note_date:
            return None, "metastasis_date_after_source_note"

    ok, veto_reason = screen_quote(quote)
    if not ok:
        return None, veto_reason

    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[:MAX_QUOTE_CHARS].rstrip() + "..."
    return {
        "evidence_type": evidence_type,
        "assertion_type": assertion_type,
        "met_site": met_site,
        "modality": modality,
        "confidence": confidence,
        "quote": quote,
        "source_note_date": source_note_date,
        "stated_metastasis_date": stated_iso,
    }, None


def validate_map_result(result, chunk):
    """Validate one chunk map, keeping every candidate that passes.

    Returns (normalized_result, fatal_error). Structural problems are fatal
    because the response is unusable. Individual bad candidates are dropped and
    tallied rather than discarding the chunk's valid candidates alongside them --
    at temperature 0 a rejected chunk would fail identically on every retry, so
    an all-or-nothing validator would lose the whole chunk permanently.
    """
    if not isinstance(result, dict):
        return None, f"non_dict_response:{type(result).__name__}"
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return None, "missing_candidates"

    rejected = Counter()
    rejected_items = []

    def record_rejection(reason, item):
        rejected[reason] += 1
        rejected_items.append({"reason": reason, "item": item})

    validated = []
    seen = set()
    for item in candidates:
        normalized, error = _validate_common_fields(item, chunk)
        if error:
            record_rejection(error, item)
            continue
        # Exact copy-forward statements add payload without adding evidence.
        key = (
            normalized["evidence_type"],
            normalized["source_note_date"],
            normalized["quote"].casefold(),
            normalized["stated_metastasis_date"],
        )
        if key in seen:
            continue
        seen.add(key)
        validated.append((normalized, item))

    kept = [normalized for normalized, _ in validated[:MAX_CANDIDATES_PER_CHUNK]]
    for _, original in validated[MAX_CANDIDATES_PER_CHUNK:]:
        record_rejection("candidates_over_cap", original)
    return (
        {
            "candidates": kept,
            "rejected": dict(rejected),
            "rejected_items": rejected_items,
        },
        None,
    )


def validate_synthesis_result(result, notes, candidate_index):
    """Validate a patient adjudication.

    A true label must survive the same gauntlet as a map candidate AND cite a
    quote that came from a validated candidate. Any failure downgrades the
    patient to an adjudicated negative with the reason recorded -- never a
    silent drop, and never a positive on evidence the model invented.
    """
    if not isinstance(result, dict):
        return None, f"non_dict_response:{type(result).__name__}"
    has_met = result.get("has_metastatic_disease")
    if not isinstance(has_met, bool):
        # No truthiness coercion of "yes"/1: an ambiguous label is unusable.
        return None, f"invalid_has_metastatic_disease:{has_met}"

    rejected = Counter()
    rejected_items = []

    def record_rejection(reason, item):
        rejected[reason] += 1
        rejected_items.append({"reason": reason, "item": item})

    rationale = _normalized_text(result.get("rationale")) or None

    if not has_met:
        evidence_fields = (
            "evidence_type", "assertion_type", "met_site", "metastasis_date",
            "source_note_date", "modality", "supporting_quote",
        )
        if any(
            result.get(field) not in (None, "", "null", "None")
            for field in evidence_fields
        ):
            # A negative label carrying stray evidence fields is still a
            # negative label; record the inconsistency and keep the verdict.
            record_rejection("negative_with_evidence_fields", result)
        return {
            "has_metastatic_disease": False,
            "finding": None,
            "rationale": rationale,
            "rejected": dict(rejected),
            "rejected_items": rejected_items,
        }, None

    # The synthesis prompt names the quote and date fields differently from the
    # map stage; normalize before running the shared validator.
    item = dict(result)
    item["quote"] = result.get("supporting_quote")
    item["stated_metastasis_date"] = result.get("metastasis_date")
    normalized, error = _validate_common_fields(item, notes)
    if error is None:
        key = (normalized["quote"].casefold(), normalized["source_note_date"])
        if key not in candidate_index:
            error = "quote_not_from_candidate"
    if error:
        record_rejection(error, result)
        return {
            "has_metastatic_disease": False,
            "finding": None,
            "rationale": rationale,
            "rejected": dict(rejected),
            "rejected_items": rejected_items,
        }, None

    return {
        "has_metastatic_disease": True,
        "finding": normalized,
        "rationale": rationale,
        "rejected": dict(rejected),
        "rejected_items": rejected_items,
    }, None


def _call_json(provider, client, model, max_retries, messages):
    response_text, error = provider.call_with_retry(client, model, messages, max_retries)
    if error:
        return None, error
    try:
        return parse_json_response(response_text), None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"json_parse:{exc}"


def _extract_chunk(provider, client, model, max_retries, mrn, chunk_index, chunk):
    payload = {
        "patient_mrn": int(mrn),
        "chunk_index": int(chunk_index),
        "notes": [
            {
                "note_date": row["note_date"],
                "note_type": row["note_type"],
                "note_text": row["snippet"],
            }
            for row in chunk
        ],
    }
    messages = [
        {"role": "system", "content": MET_DX_MAP_PROMPT + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    result, error = _call_json(provider, client, model, max_retries, messages)
    if error:
        return None, error
    return validate_map_result(result, chunk)


def extract_patient(provider, client, model, max_retries, mrn, indexed_chunks):
    """Map outstanding chunks, preserving their original evidence indices."""
    outputs = []
    chunk_results = []
    for chunk_index, chunk in indexed_chunks:
        result, error = _extract_chunk(
            provider, client, model, max_retries, mrn, chunk_index, chunk
        )
        if error:
            chunk_results.append(
                {
                    "chunk_index": chunk_index,
                    "num_candidates": 0,
                    "num_dropped": 0,
                    "status": error,
                    "result_json": None,
                }
            )
            continue
        outputs.append((chunk_index, result))
        chunk_results.append(
            {
                "chunk_index": chunk_index,
                "num_candidates": len(result["candidates"]),
                "num_dropped": sum(result.get("rejected", {}).values()),
                "status": "ok_with_rejections" if result.get("rejected") else "ok",
                "result_json": json.dumps(result, ensure_ascii=False, sort_keys=True),
            }
        )
    return outputs, chunk_results


def _flatten_notes(chunks):
    """Flatten {chunk_index: [note, ...]} to the list find_quote_support takes."""
    return [note for _, notes in sorted(chunks.items()) for note in notes]


def _candidate_index(chunk_outputs):
    """Set of (quote_casefold, source_note_date) pairs the map stage validated."""
    index = set()
    for result in chunk_outputs.values():
        for candidate in result.get("candidates", []):
            index.add(
                (
                    _normalized_text(candidate.get("quote")).casefold(),
                    candidate.get("source_note_date"),
                )
            )
    return index


def synthesize_patient(provider, client, model, max_retries, mrn, chunk_outputs, chunks):
    payload = {
        "patient_mrn": int(mrn),
        "candidates": [
            # `rejected` is local validation bookkeeping, not evidence -- forwarding
            # it would put our own error taxonomy in the model's context for no gain.
            {"chunk_index": index, **candidate}
            for index, result in sorted(chunk_outputs.items())
            for candidate in result.get("candidates", [])
        ],
    }
    messages = [
        {"role": "system", "content": MET_DX_SYNTHESIS_PROMPT + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    result, error = _call_json(provider, client, model, max_retries, messages)
    if error:
        return None, error
    return validate_synthesis_result(
        result, _flatten_notes(chunks), _candidate_index(chunk_outputs)
    )


def _load_chunk_outputs(path, run_config):
    outputs = {}
    if not path.exists() or path.stat().st_size == 0:
        return outputs
    log = pl.read_parquet(path)
    for row in log.iter_rows(named=True):
        if row.get("status") not in SUCCESS_STATUSES or row.get("run_config") != run_config:
            continue
        mrn = _to_exact_int(row.get("DFCI_MRN"))
        idx = _to_exact_int(row.get("chunk_index"))
        if mrn is None or idx is None or not row.get("result_json"):
            continue
        try:
            result = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        outputs.setdefault(mrn, {})[idx] = result
    return outputs


def _load_chunk_statuses(path, run_config):
    statuses = {}
    if not path.exists() or path.stat().st_size == 0:
        return statuses
    log = pl.read_parquet(path)
    for row in log.iter_rows(named=True):
        if row.get("run_config") != run_config:
            continue
        mrn = _to_exact_int(row.get("DFCI_MRN"))
        idx = _to_exact_int(row.get("chunk_index"))
        if mrn is not None and idx is not None:
            statuses.setdefault(mrn, {})[idx] = row.get("status") or "unknown"
    return statuses


def _load_completed_syntheses(path, run_config):
    completed = {}
    if not path.exists() or path.stat().st_size == 0:
        return completed
    log = pl.read_parquet(path)
    for row in log.iter_rows(named=True):
        if row.get("status") not in SUCCESS_STATUSES or row.get("run_config") != run_config:
            continue
        mrn = _to_exact_int(row.get("DFCI_MRN"))
        if mrn is None or not row.get("result_json"):
            continue
        try:
            completed[mrn] = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            continue
    return completed


def rebuild_rejected_findings(
    chunk_log_path, processed_path, rejected_path, run_config
):
    """Materialize every rejected object for audit/recovery.

    This file is the tuning signal for veto.py: histogram its `reason` column
    to see which cues are firing and spot-check them for over-rejection.
    """
    rows = []

    def add_rejections(mrn, stage, chunk_index, result):
        for entry in result.get("rejected_items", []):
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "DFCI_MRN": mrn,
                    "stage": stage,
                    "chunk_index": chunk_index,
                    "reason": entry.get("reason"),
                    "item_json": json.dumps(
                        entry.get("item"), ensure_ascii=False, sort_keys=True
                    ),
                    "run_config": run_config,
                }
            )

    if chunk_log_path.exists() and chunk_log_path.stat().st_size > 0:
        log = pl.read_parquet(chunk_log_path)
        for row in log.iter_rows(named=True):
            if row.get("run_config") != run_config or not row.get("result_json"):
                continue
            mrn = _to_exact_int(row.get("DFCI_MRN"))
            chunk_index = _to_exact_int(row.get("chunk_index"))
            if mrn is None or chunk_index is None:
                continue
            try:
                result = json.loads(row["result_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            add_rejections(mrn, "map", chunk_index, result)

    if processed_path.exists() and processed_path.stat().st_size > 0:
        log = pl.read_parquet(processed_path)
        for row in log.iter_rows(named=True):
            if row.get("run_config") != run_config or not row.get("result_json"):
                continue
            mrn = _to_exact_int(row.get("DFCI_MRN"))
            if mrn is None:
                continue
            try:
                result = json.loads(row["result_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            add_rejections(mrn, "synthesis", None, result)

    _write_rows_atomic(rejected_path, rows, REJECTED_COLUMNS)
    return len(rows)


def rebuild_candidates(chunk_log_path, candidates_path, run_config):
    """Materialize every validated map candidate for audit."""
    rows = []
    chunk_outputs = _load_chunk_outputs(chunk_log_path, run_config)
    for mrn in sorted(chunk_outputs):
        for chunk_index in sorted(chunk_outputs[mrn]):
            for candidate in chunk_outputs[mrn][chunk_index].get("candidates", []):
                rows.append(
                    {
                        "DFCI_MRN": mrn,
                        "chunk_index": chunk_index,
                        "source_note_date": candidate.get("source_note_date"),
                        "evidence_type": candidate.get("evidence_type"),
                        "assertion_type": candidate.get("assertion_type"),
                        "met_site": candidate.get("met_site"),
                        "stated_metastasis_date": candidate.get("stated_metastasis_date"),
                        "modality": candidate.get("modality"),
                        "quote": candidate.get("quote"),
                        "confidence": candidate.get("confidence"),
                    }
                )
    _write_rows_atomic(candidates_path, rows, CANDIDATE_COLUMNS)
    return len(rows)


def _earliest_qualifying_note_date(chunk_outputs):
    """Earliest source note date among ALL validated candidates for the patient.

    This is the common path, not a fallback edge case: metastatic disease is
    rarely given an explicit date in the text, so most positives resolve their
    date here.

    Deliberately broader than the NEPC task's same-evidence_type version. The
    question asked is "when was metastatic disease FIRST mentioned", and the
    same underlying fact is routinely first written by a radiologist
    ("imaging_metastasis") and only later restated by an oncologist
    ("stated_metastatic_disease"). Keying the earliest date to the adjudicated
    candidate's own type would date the patient to the restatement and lose the
    original documentation, which is the answer the task is for. Every
    candidate considered here has already cleared grounding and the veto, so
    "earliest" cannot be pulled backwards by a negated or hedged mention.

    The supporting quote still comes from the adjudicated candidate, and
    `date_source` stays "note_date", so this fallback remains filterable
    downstream.
    """
    dates = [
        candidate.get("source_note_date")
        for result in chunk_outputs.values()
        for candidate in result.get("candidates", [])
        if candidate.get("source_note_date")
    ]
    return min(dates) if dates else None


def build_labels(
    processed_path,
    chunk_log_path,
    labels_path,
    run_config,
    patient_chunks,
    auto_negative_mrns=None,
):
    """Build one label row per patient from saved adjudications."""
    auto_negative_mrns = {int(mrn) for mrn in (auto_negative_mrns or [])}
    completed = _load_completed_syntheses(processed_path, run_config)
    chunk_outputs = _load_chunk_outputs(chunk_log_path, run_config)

    rows = []
    for mrn in sorted(completed):
        if mrn not in patient_chunks:
            continue
        result = completed[mrn]
        outputs = chunk_outputs.get(mrn, {})
        num_candidates = sum(
            len(chunk.get("candidates", [])) for chunk in outputs.values()
        )
        finding = result.get("finding")
        if not result.get("has_metastatic_disease") or not isinstance(finding, dict):
            rows.append(
                {
                    "DFCI_MRN": mrn,
                    "has_metastatic_disease": False,
                    "first_metastasis_date": None,
                    "date_source": None,
                    "date_precision": "unknown",
                    "stated_metastasis_date": None,
                    "source_note_date": None,
                    "evidence_type": None,
                    "assertion_type": None,
                    "met_site": None,
                    "modality": None,
                    "supporting_quote": None,
                    "confidence": None,
                    "rationale": result.get("rationale"),
                    "num_candidates": num_candidates,
                    "label_source": "adjudicated",
                }
            )
            continue

        stated = finding.get("stated_metastasis_date")
        source_note_date = finding.get("source_note_date")
        event_date, date_source, date_precision = resolve_date(
            stated, source_note_date
        )
        if date_source == "note_date":
            # An undated positive is kept, not dropped; date_source stays
            # "note_date" so the fallback remains filterable.
            earliest = _earliest_qualifying_note_date(outputs)
            if earliest and (event_date is None or earliest < event_date):
                event_date = earliest
        rows.append(
            {
                "DFCI_MRN": mrn,
                "has_metastatic_disease": True,
                "first_metastasis_date": event_date,
                "date_source": date_source,
                "date_precision": date_precision,
                "stated_metastasis_date": stated,
                "source_note_date": source_note_date,
                "evidence_type": finding.get("evidence_type"),
                "assertion_type": finding.get("assertion_type"),
                "met_site": finding.get("met_site"),
                "modality": finding.get("modality"),
                "supporting_quote": finding.get("quote"),
                "confidence": finding.get("confidence"),
                "rationale": result.get("rationale"),
                "num_candidates": num_candidates,
                "label_source": "adjudicated",
            }
        )

    # Cohort patients whose notes produced no trigger-bearing evidence at all.
    # Materializing them keeps the denominator explicit rather than implied by
    # absence from the file.
    labeled = {row["DFCI_MRN"] for row in rows}
    for mrn in sorted(auto_negative_mrns - labeled):
        rows.append(
            {
                "DFCI_MRN": mrn,
                "has_metastatic_disease": False,
                "first_metastasis_date": None,
                "date_source": None,
                "date_precision": "unknown",
                "stated_metastasis_date": None,
                "source_note_date": None,
                "evidence_type": None,
                "assertion_type": None,
                "met_site": None,
                "modality": None,
                "supporting_quote": None,
                "confidence": None,
                "rationale": None,
                "num_candidates": 0,
                "label_source": "auto_negative_no_evidence",
            }
        )
    rows.sort(key=lambda row: row["DFCI_MRN"])
    _write_rows_atomic(labels_path, rows, LABEL_COLUMNS)
    return len(rows)


def rebuild_final_artifacts(
    *,
    processed_path,
    chunk_log_path,
    candidates_path,
    rejected_path,
    labels_path,
    run_config,
    patient_chunks,
    selected_mrns,
    evidence_meta,
    limited_run=False,
):
    """Rebuild derived outputs from durable state without invoking a provider."""
    candidate_count = rebuild_candidates(chunk_log_path, candidates_path, run_config)
    rejected_count = rebuild_rejected_findings(
        chunk_log_path, processed_path, rejected_path, run_config
    )
    recorded_cohort = {
        mrn
        for value in (evidence_meta.get("cohort_mrns") or [])
        if (mrn := _to_exact_int(value)) is not None
    }
    cohort_mrns = set(selected_mrns) if selected_mrns is not None else recorded_cohort
    auto_negative_mrns = set() if limited_run else cohort_mrns - set(patient_chunks)
    label_count = build_labels(
        processed_path,
        chunk_log_path,
        labels_path,
        run_config,
        patient_chunks,
        auto_negative_mrns=auto_negative_mrns,
    )
    return candidate_count, rejected_count, label_count


def _load_patient_chunks(evidence_df):
    required = {"DFCI_MRN", "chunk_index", "note_date", "note_type", "snippet"}
    missing = sorted(required - set(evidence_df.columns))
    if missing:
        raise ValueError(f"Evidence table missing required columns: {missing}")
    patient_chunks = {}
    invalid = 0
    for row in evidence_df.iter_rows(named=True):
        mrn = _to_exact_int(row.get("DFCI_MRN"))
        chunk_index = _to_exact_int(row.get("chunk_index"))
        if (
            mrn is None
            or chunk_index is None
            or not 0 <= chunk_index <= MAX_CHUNK_INDEX
            or not _normalized_text(row.get("snippet"))
        ):
            invalid += 1
            continue
        patient_chunks.setdefault(mrn, {}).setdefault(chunk_index, []).append(
            {
                "note_date": to_iso_date(row.get("note_date")),
                "note_type": row.get("note_type") or "Unknown",
                "snippet": row.get("snippet") or "",
            }
        )
    if invalid:
        print(f"  Skipped {invalid} invalid evidence rows")
    return patient_chunks


def run(args):
    if args.max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if args.max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if args.limit_patients is not None and args.limit_patients < 0:
        raise ValueError("limit_patients must be >= 0")
    if getattr(args, "rebuild_labels_only", False) and args.overwrite:
        raise ValueError(
            "--rebuild-labels-only cannot be combined with --overwrite because "
            "overwrite deletes the saved adjudication state needed for an offline rebuild."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "met_dx_evidence.parquet")
    meta_path = getattr(args, "evidence_meta_path", None) or _meta_path_for_evidence(
        evidence_path
    )
    candidates_path = args.output_dir / "met_dx_candidates_raw.parquet"
    chunk_log_path = args.output_dir / "met_dx_processed_chunks.parquet"
    processed_path = args.output_dir / "met_dx_processed_patients.parquet"
    labels_path = args.output_dir / "met_dx_labels.parquet"
    rejected_path = args.output_dir / "met_dx_rejected_findings.parquet"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_met_dx_notes.py first."
        )
    if args.overwrite:
        for path in (
            candidates_path,
            chunk_log_path,
            processed_path,
            labels_path,
            rejected_path,
        ):
            path.unlink(missing_ok=True)

    provider = get_provider(args.provider)
    model = args.model or provider.default_model
    scan_config, run_config = check_resume_config(
        chunk_log_path, meta_path, args.provider, model, evidence_path
    )
    evidence_meta = read_scan_config_meta(meta_path) or {}
    print(f"Extraction fingerprint: {run_config} ({args.provider}/{model}, {VETO_VERSION})")

    evidence_df = pl.read_parquet(evidence_path)
    patient_chunks = _load_patient_chunks(evidence_df)
    print(
        f"Loaded evidence: {evidence_df.height} rows for {len(patient_chunks)} patients "
        f"({sum(len(chunks) for chunks in patient_chunks.values())} chunks)"
    )

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    target_mrns = [
        mrn for mrn in sorted(patient_chunks)
        if selected_mrns is None or mrn in selected_mrns
    ]
    if args.limit_patients is not None:
        target_mrns = target_mrns[: args.limit_patients]

    if getattr(args, "rebuild_labels_only", False):
        candidate_count, rejected_count, label_count = rebuild_final_artifacts(
            processed_path=processed_path,
            chunk_log_path=chunk_log_path,
            candidates_path=candidates_path,
            rejected_path=rejected_path,
            labels_path=labels_path,
            run_config=run_config,
            patient_chunks=patient_chunks,
            selected_mrns=selected_mrns,
            evidence_meta=evidence_meta,
            limited_run=args.limit_patients is not None,
        )
        print("Offline rebuild only: no LLM calls were made.")
        print(f"Wrote validated candidates ({candidate_count} rows): {candidates_path}")
        print(f"Wrote rejected-finding audit ({rejected_count} rows): {rejected_path}")
        print(f"Wrote metastasis labels ({label_count} rows): {labels_path}")
        return

    done_chunks = read_done_chunks(chunk_log_path, run_config)
    map_todo = []
    for mrn in target_mrns:
        outstanding = [
            (chunk_index, chunk)
            for chunk_index, chunk in sorted(patient_chunks[mrn].items())
            if (mrn, chunk_index) not in done_chunks
        ]
        if outstanding:
            map_todo.append((mrn, outstanding))
    print(
        f"Chunk mapping: {len(map_todo)} patients, "
        f"{sum(len(items) for _, items in map_todo)} outstanding calls"
    )

    # Findings this run's LLM calls produced but validation refused, by reason.
    # Only calls made *this* run contribute; resumed chunks keep their counts in
    # the chunk log's num_dropped rather than being re-tallied here.
    run_rejected = Counter()

    client = None
    if map_todo:
        client = provider.build_client()

        def map_worker(mrn, indexed_chunks):
            outputs, results = extract_patient(
                provider, client, model, args.max_retries, mrn, indexed_chunks
            )
            return mrn, outputs, results

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(map_worker, mrn, chunks): (mrn, chunks)
                for mrn, chunks in map_todo
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Map", unit="pt"):
                mrn, attempted = futures[future]
                try:
                    _, _, results = future.result()
                except Exception as exc:  # noqa: BLE001
                    results = [
                        {
                            "chunk_index": index,
                            "num_candidates": 0,
                            "num_dropped": 0,
                            "status": f"worker_error:{type(exc).__name__}:{str(exc)[:160]}",
                            "result_json": None,
                        }
                        for index, _ in attempted
                    ]
                for result in results:
                    if result.get("result_json"):
                        try:
                            run_rejected.update(
                                json.loads(result["result_json"]).get("rejected", {})
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass
                append_rows(
                    chunk_log_path,
                    [
                        {
                            "DFCI_MRN": mrn,
                            "scan_config": scan_config,
                            "run_config": run_config,
                            **result,
                        }
                        for result in results
                    ],
                    CHUNK_COLUMNS,
                )
    compact_log(chunk_log_path, CHUNK_COLUMNS, ["DFCI_MRN", "chunk_index"])

    chunk_outputs = _load_chunk_outputs(chunk_log_path, run_config)
    chunk_statuses = _load_chunk_statuses(chunk_log_path, run_config)
    completed_syntheses = _load_completed_syntheses(processed_path, run_config)
    synthesis_todo = []
    for mrn in target_mrns:
        expected = set(patient_chunks[mrn])
        available = set(chunk_outputs.get(mrn, {}))
        if expected == available and mrn not in completed_syntheses:
            synthesis_todo.append(mrn)
        elif expected != available:
            failed_statuses = [
                status
                for index, status in sorted(chunk_statuses.get(mrn, {}).items())
                if index in expected and status not in SUCCESS_STATUSES
            ]
            status = (
                f"partial_map:{len(available)}/{len(expected)}:"
                f"{failed_statuses[0] if failed_statuses else 'missing_chunk_output'}"
            )
            map_rejected = Counter()
            for chunk_result in chunk_outputs.get(mrn, {}).values():
                map_rejected.update(chunk_result.get("rejected", {}))
            append_rows(
                processed_path,
                [
                    {
                        "DFCI_MRN": mrn,
                        "num_chunks": len(expected),
                        "num_chunks_ok": len(available),
                        "num_candidates": 0,
                        "has_metastatic_disease": None,
                        "num_dropped": sum(map_rejected.values()),
                        "status": status,
                        "run_config": run_config,
                        "result_json": None,
                    }
                ],
                PROCESSED_COLUMNS,
            )
    print(f"Patient adjudication: {len(synthesis_todo)} outstanding calls")

    if synthesis_todo:
        if client is None:
            client = provider.build_client()

        def synthesis_worker(mrn):
            result, error = synthesize_patient(
                provider,
                client,
                model,
                args.max_retries,
                mrn,
                chunk_outputs[mrn],
                patient_chunks[mrn],
            )
            return mrn, result, error

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(synthesis_worker, mrn): mrn for mrn in synthesis_todo
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Adjudicate", unit="pt"
            ):
                mrn = futures[future]
                try:
                    _, result, error = future.result()
                except Exception as exc:  # noqa: BLE001
                    result, error = None, f"worker_error:{type(exc).__name__}:{str(exc)[:160]}"
                n_chunks = len(patient_chunks[mrn])
                synthesis_rejected = Counter(result.get("rejected", {})) if result else Counter()
                map_rejected = Counter()
                for chunk_result in chunk_outputs.get(mrn, {}).values():
                    map_rejected.update(chunk_result.get("rejected", {}))
                run_rejected.update(synthesis_rejected)
                num_candidates = sum(
                    len(chunk.get("candidates", []))
                    for chunk in chunk_outputs.get(mrn, {}).values()
                )
                append_rows(
                    processed_path,
                    [
                        {
                            "DFCI_MRN": mrn,
                            "num_chunks": n_chunks,
                            "num_chunks_ok": n_chunks,
                            "num_candidates": num_candidates,
                            "has_metastatic_disease": (
                                result.get("has_metastatic_disease")
                                if result is not None
                                else None
                            ),
                            "num_dropped": sum(synthesis_rejected.values())
                            + sum(map_rejected.values()),
                            "status": (
                                "ok_with_rejections"
                                if result is not None
                                and (synthesis_rejected or map_rejected)
                                else ("ok" if result is not None else error)
                            ),
                            "run_config": run_config,
                            "result_json": (
                                json.dumps(result, ensure_ascii=False, sort_keys=True)
                                if result is not None
                                else None
                            ),
                        }
                    ],
                    PROCESSED_COLUMNS,
                )
    compact_log(processed_path, PROCESSED_COLUMNS, ["DFCI_MRN"])

    if run_rejected:
        total = sum(run_rejected.values())
        print(f"Rejected {total} findings during validation, by reason:")
        for reason, count in run_rejected.most_common():
            print(f"  {count:>6}  {reason}")

    candidate_count, rejected_count, label_count = rebuild_final_artifacts(
        processed_path=processed_path,
        chunk_log_path=chunk_log_path,
        candidates_path=candidates_path,
        rejected_path=rejected_path,
        labels_path=labels_path,
        run_config=run_config,
        patient_chunks=patient_chunks,
        selected_mrns=selected_mrns,
        evidence_meta=evidence_meta,
        limited_run=args.limit_patients is not None,
    )
    print(f"Wrote validated candidates ({candidate_count} rows): {candidates_path}")
    print(f"Wrote rejected-finding audit ({rejected_count} rows): {rejected_path}")
    print(f"Wrote metastasis labels ({label_count} rows): {labels_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
