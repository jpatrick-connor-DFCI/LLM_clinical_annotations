"""Build a longitudinal canonical Aparicio AVPC/NEPC timeline.

Stage 1 (`preprocessing/cli/collect_nepc_notes.py`) writes trigger-bearing
evidence chunks. This stage maps every chunk into compact structured evidence,
then reduces all successful chunk maps for a patient into one patient-level
criterion result. Chunk maps and patient reductions are durable resume state.

The resume fingerprint includes the evidence scan, provider, model, prompt
texts, and prompt schema version. Changing any of them requires `--overwrite`;
this prevents mixed-model or mixed-prompt research outputs.
"""

import argparse
import calendar
import hashlib
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import CLINICAL_SAFETY_CONTEXT, DEFAULT_DATA_PATH  # noqa: E402
from preprocessing.longitudinal import (  # noqa: E402
    flatten_ws,
    file_sha256,
    parse_stated_date,
    read_scan_config_meta,
    resolve_date,
)
from preprocessing.notes import load_selected_mrns, to_iso_date  # noqa: E402
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.longitudinal_NEPC.prompts import (  # noqa: E402
    NEPC_SYNTHESIS_PROMPT,
    NEPC_SYSTEM_PROMPT,
    PROMPT_SCHEMA_VERSION,
)

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_avpc_nepc_timeline"

CRITERION_LABELS = {
    "C1": "histologic small-cell prostate carcinoma",
    "C2": "exclusively visceral metastases",
    "C3": "predominantly lytic bone metastases",
    "C4": "bulky nodal disease or >=5 cm high-grade prostate/pelvic mass",
    "C5": "PSA <=10 ng/mL plus >=20 bone metastases",
    "C6": "neuroendocrine marker plus qualifying LDH/CEA/hypercalcemia",
    "C7": "castration-resistant progression within <=6 months of hormonal therapy",
    "NEPC:small_cell_dx": "NEPC: neuroendocrine or small-cell prostate carcinoma",
    "NEPC:histologic_transformation": "NEPC: histologic transformation from adenocarcinoma",
    "NEPC:ne_features": "NEPC: neuroendocrine features / differentiation",
    "NEPC:positive_ne_ihc": "NEPC: positive neuroendocrine IHC on prostate-derived specimen",
}
VALID_CRITERIA = set(CRITERION_LABELS)
VALID_MODALITIES = {"pathology", "imaging", "clinical", "labs"}
VALID_CONFIDENCE = {"high", "medium", "low"}
SUCCESS_STATUSES = {"ok", "ok_with_rejections"}
MAX_QUOTE_CHARS = 1000
MAX_CHUNK_INDEX = 10_000
MAX_EVIDENCE_ITEMS_PER_CHUNK = 30

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

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "event_date",
    "date_source",
    "date_precision",
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

CHUNK_COLUMNS = [
    "DFCI_MRN",
    "chunk_index",
    "num_criteria",
    "num_evidence_items",
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
    "num_criteria",
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

VALID_FACT_TYPES_BY_CRITERION = {
    "C1": {"small_cell_histology"},
    "C2": {
        "visceral_site",
        "bone_metastasis_status",
        "non_visceral_metastasis_status",
    },
    "C3": {"lytic_bone_pattern"},
    "C4": {
        "bulky_nodal_measurement",
        "prostate_pelvic_mass_measurement",
        "gleason_score",
    },
    "C5": {"psa_value", "disease_context", "bone_metastasis_count"},
    "C6": {
        "neuroendocrine_marker",
        "ldh_value_and_uln",
        "cea_value_and_uln",
        "malignant_hypercalcemia",
        "alternative_cause_assessment",
    },
    "C7": {"hormonal_therapy_start", "crpc_progression"},
    "NEPC:small_cell_dx": {"small_cell_diagnosis"},
    "NEPC:histologic_transformation": {"histologic_transformation"},
    "NEPC:ne_features": {"neuroendocrine_features"},
    "NEPC:positive_ne_ihc": {"positive_ne_ihc"},
}

CRITERION_QUOTE_PATTERNS = {
    "C1": r"\bsmall[\s-]?cell\b",
    "C2": (
        r"\b(?:visceral|liver|hepatic|lung|pulmonary|adrenal|brain|"
        r"pleur\w*|peritone\w*)\b"
    ),
    "C3": r"\b(?:lytic|destructive\s+bone)\b",
    "C4": (
        r"\b(?:bulky|lymphadenopathy|adenopathy|nodal|prostat\w*|pelvic|gleason)\b"
        r"|(?:\b(?:[5-9]|[1-9]\d+)(?:\.\d+)?\s*cm\b)"
    ),
    "C5": r"\b(?:psa|bone|osseous|metasta\w*)\b",
    "C6": (
        r"\b(?:neuroendocrine|synaptophysin|chromogranin|grp|ldh|cea|"
        r"hypercalc\w*)\b"
    ),
    "C7": (
        r"\b(?:castration[\s-]?resistant|androgen[\s-]?independent|"
        r"androgen\s+deprivation|adt|hormonal\s+therapy|progress\w*)\b"
    ),
    "NEPC:small_cell_dx": r"\b(?:small[\s-]?cell|neuroendocrine\s+carcinoma)\b",
    "NEPC:histologic_transformation": (
        r"\b(?:transform\w*|transdifferentiat\w*|treatment[\s-]?emergent)\b"
    ),
    "NEPC:ne_features": r"\bneuroendocrine\s+(?:feature\w*|differentiat\w*)\b",
    "NEPC:positive_ne_ihc": (
        r"\b(?:synaptophysin|chromogranin|cd56|nse|insm1|"
        r"neuron[\s-]specific\s+enolase)\b"
    ),
}

FACT_TYPE_QUOTE_PATTERNS = {
    "small_cell_histology": CRITERION_QUOTE_PATTERNS["C1"],
    "visceral_site": CRITERION_QUOTE_PATTERNS["C2"],
    "bone_metastasis_status": r"\b(?:bone|osseous|skeletal)\b",
    "non_visceral_metastasis_status": (
        r"\b(?:bone|osseous|skeletal|nodal|lymph\s+node|soft[\s-]?tissue)\b"
    ),
    "lytic_bone_pattern": CRITERION_QUOTE_PATTERNS["C3"],
    "bulky_nodal_measurement": (
        r"\b(?:bulky|lymphadenopathy|adenopathy|nodal|lymph\s+node)\b"
        r"|(?:\b\d+(?:\.\d+)?\s*(?:cm|mm)\b)"
    ),
    "prostate_pelvic_mass_measurement": (
        r"\b(?:prostat\w*|pelvic|mass)\b|(?:\b\d+(?:\.\d+)?\s*(?:cm|mm)\b)"
    ),
    "gleason_score": r"\bgleason\b",
    "psa_value": r"\bpsa\b",
    "disease_context": (
        r"\b(?:initial\s+presentation|symptomatic|castration[\s-]?resistant|"
        r"crpc|progress\w*|before\s+(?:adt|androgen\s+deprivation))\b"
    ),
    "bone_metastasis_count": r"\b(?:bone|osseous|skeletal|lesion\w*)\b",
    "neuroendocrine_marker": (
        r"\b(?:neuroendocrine|synaptophysin|chromogranin|grp)\b"
    ),
    "ldh_value_and_uln": r"\b(?:ldh|lactate\s+dehydrogenase)\b",
    "cea_value_and_uln": r"\b(?:cea|carcinoembryonic\s+antigen)\b",
    "malignant_hypercalcemia": r"\b(?:hypercalc\w*|calcium)\b",
    "alternative_cause_assessment": (
        r"\b(?:cause|caused|due\s+to|attribut\w*|explain\w*|etiolog\w*)\b"
    ),
    "hormonal_therapy_start": (
        r"\b(?:androgen\s+deprivation|adt|hormonal\s+therapy|leuprolide|"
        r"lupron|degarelix|firmagon|relugolix|orgovyx|orchiectomy)\b"
    ),
    "crpc_progression": (
        r"\b(?:castration[\s-]?resistant|androgen[\s-]?independent|crpc|"
        r"progress\w*)\b"
    ),
    "small_cell_diagnosis": CRITERION_QUOTE_PATTERNS["NEPC:small_cell_dx"],
    "histologic_transformation": CRITERION_QUOTE_PATTERNS[
        "NEPC:histologic_transformation"
    ],
    "neuroendocrine_features": CRITERION_QUOTE_PATTERNS["NEPC:ne_features"],
    "positive_ne_ihc": CRITERION_QUOTE_PATTERNS["NEPC:positive_ne_ihc"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map/reduce AVPC/NEPC evidence into a longitudinal criteria timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-path", type=Path, default=None)
    parser.add_argument(
        "--evidence-meta-path",
        type=Path,
        default=None,
        help="Override evidence metadata path. By default it is derived from --evidence-path.",
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--provider", choices=["dfci_gpt", "vertex_ai"], default="dfci_gpt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-patients", type=int, default=None)
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
    """Append complete TSV records; callers compact them atomically afterward."""
    if not rows:
        return
    df = pl.DataFrame({c: [r.get(c) for r in rows] for c in columns})
    write_header = not path.exists() or path.stat().st_size == 0
    text = df.write_csv(separator="\t", include_header=write_header)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def _write_rows_atomic(path, rows, columns):
    if rows:
        df = pl.DataFrame({c: [row.get(c) for row in rows] for c in columns})
    else:
        df = pl.DataFrame(schema={c: pl.Utf8 for c in columns})
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.write_csv(tmp_path, separator="\t")
    tmp_path.replace(path)


def compact_log(path, columns, key_columns):
    if not path.exists() or path.stat().st_size == 0:
        return
    log = pl.read_csv(path, separator="\t", infer_schema_length=0)
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
    name = evidence_path.name
    if name.endswith(".tsv"):
        name = f"{name[:-4]}.meta.json"
    else:
        name = f"{name}.meta.json"
    return evidence_path.with_name(name)


def extraction_run_config(scan_config, provider_name, model):
    payload = {
        "schema": PROMPT_SCHEMA_VERSION,
        "scan_config": scan_config,
        "provider": provider_name,
        "model": model,
        "map_prompt": NEPC_SYSTEM_PROMPT,
        "synthesis_prompt": NEPC_SYNTHESIS_PROMPT,
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

    log = pl.read_csv(chunk_log_path, separator="\t", infer_schema_length=0)
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
            "Provider, model, prompt, schema, or evidence changed since the existing "
            "chunk log was created. Re-run with --overwrite to avoid mixed outputs."
        )
    return scan_config, run_config


def read_done_chunks(path, run_config=None):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    log = pl.read_csv(path, separator="\t", infer_schema_length=0)
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


def normalize_criterion(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in VALID_CRITERIA:
        return text
    upper = text.upper()
    if upper in VALID_CRITERIA:
        return upper
    if ":" in text:
        prefix, _, suffix = text.partition(":")
        candidate = f"{prefix.strip().upper()}:{suffix.strip()}"
        if candidate in VALID_CRITERIA:
            return candidate
    return None


def _normalized_text(value):
    return flatten_ws(value) or ""


def _quote_core(quote):
    """Strip decoration models add around otherwise-verbatim quotes.

    Leading/trailing ellipses and wrapping quotation marks are the common ways a
    model signals "excerpt" while still copying the underlying text exactly;
    treating them as mismatches would reject genuinely grounded findings. Curly
    quotes/dashes are folded to ASCII for the same reason.
    """
    text = quote.strip()
    for pair in (("“", "”"), ("‘", "’"), ('"', '"'), ("'", "'")):
        if len(text) >= 2 and text.startswith(pair[0]) and text.endswith(pair[1]):
            text = text[1:-1].strip()
            break
    while text.startswith(("...", "…")):
        text = text[3:] if text.startswith("...") else text[1:]
        text = text.lstrip()
    while text.endswith(("...", "…")):
        text = text[:-3] if text.endswith("...") else text[:-1]
        text = text.rstrip()
    return text


def _fold_quote_chars(text):
    """Fold unicode punctuation variants to ASCII so quoting style can't break matching."""
    for source, target in (
        ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
        ("–", "-"), ("—", "-"), ("−", "-"), (" ", " "),
    ):
        text = text.replace(source, target)
    return text


def _find_support_note(item, chunks):
    """Return `(chunk_index, note_date)` containing this item's quote, else None.

    Prefer the cited source date. If the quote is grounded in a different note,
    return that note's actual date so callers can correct provenance *before*
    validating diagnosis/fact dates. This preserves a grounded finding without
    allowing an invented future source date to legitimize an impossible event.
    """
    source_date = item.get("source_note_date")
    quote = _quote_core(_fold_quote_chars(_normalized_text(item.get("quote")))).lower()
    if not quote:
        return None
    matches = []
    for chunk_index, chunk in chunks.items():
        for note in chunk:
            snippet = _fold_quote_chars(_normalized_text(note.get("snippet"))).lower()
            if quote in snippet:
                matches.append((chunk_index, note.get("note_date")))
    if not matches:
        return None
    for chunk_index, note_date in matches:
        if note_date == source_date:
            return chunk_index, note_date
    # Identical/copy-forward text can occur on multiple dates. The earliest
    # grounded note is the conservative documented-onset provenance.
    return min(
        matches,
        key=lambda match: (match[1] or "9999-99-99", match[0]),
    )


def _find_support_chunk(item, chunks):
    """Backward-compatible wrapper returning only the grounded chunk index."""
    support = _find_support_note(item, chunks)
    if support is not None:
        return support[0]
    return None


def _quote_supports_criterion(criterion, quote):
    pattern = CRITERION_QUOTE_PATTERNS.get(criterion)
    return bool(pattern and re.search(pattern, quote, flags=re.IGNORECASE))


def _quote_supports_fact_type(fact_type, quote):
    pattern = FACT_TYPE_QUOTE_PATTERNS.get(fact_type)
    return bool(pattern and re.search(pattern, quote, flags=re.IGNORECASE))


def _validate_common_finding(item, chunks, *, date_field):
    if not isinstance(item, dict):
        return None, "finding_not_object"
    criterion = normalize_criterion(item.get("criterion") or item.get("candidate_criterion"))
    if criterion is None:
        return None, f"invalid_criterion:{item.get('criterion') or item.get('candidate_criterion')}"
    modality = item.get("modality")
    confidence = item.get("confidence")
    if modality not in VALID_MODALITIES:
        return None, f"invalid_modality:{modality}"
    if confidence not in VALID_CONFIDENCE:
        return None, f"invalid_confidence:{confidence}"
    quote = _normalized_text(item.get("quote"))
    if not quote:
        return None, "missing_quote"
    normalized = dict(item)
    normalized["quote"] = quote
    normalized["source_note_date"] = to_iso_date(item.get("source_note_date"))
    support = _find_support_note(normalized, chunks)
    if support is None:
        return None, "quote_or_source_not_in_evidence"
    support_index, actual_source_date = support
    if normalized["source_note_date"] != actual_source_date:
        normalized["_claimed_source_note_date"] = normalized["source_note_date"]
        normalized["source_note_date"] = actual_source_date
    normalized[date_field] = item.get(date_field)
    if item.get(date_field) not in (None, "", "null", "None"):
        stated_iso, _ = parse_stated_date(item.get(date_field))
        if stated_iso is None:
            return None, f"invalid_{date_field}:{item.get(date_field)}"
        source_iso = normalized["source_note_date"]
        if source_iso and stated_iso > source_iso:
            return None, f"{date_field}_after_source_note"
    if len(quote) > MAX_QUOTE_CHARS:
        normalized["quote"] = quote[:MAX_QUOTE_CHARS].rstrip() + "..."
    normalized["_support_chunk_index"] = support_index
    normalized["criterion"] = criterion
    normalized["modality"] = modality
    normalized["confidence"] = confidence
    return normalized, None


def _resolve_c2_pattern(normalized):
    """Return (visceral_met_pattern, reject_reason) for one validated finding.

    C2 is "exclusively visceral", so a finding the model itself marks as having
    concurrent bone/non-visceral disease contradicts the criterion it is claiming
    and is rejected as that single finding. Any other criterion carries "none":
    visceral_met_pattern is only meaningful for C2.
    """
    if normalized["criterion"] != "C2":
        return "none", None
    vmp = normalized.get("visceral_met_pattern")
    if vmp == "visceral_only":
        return "visceral_only", None
    if vmp in (None, "", "none"):
        # The map prompt says to use "none" when exclusivity isn't established in
        # this chunk; that is a not-yet-supported C2, not a malformed response.
        return None, "c2_without_established_exclusivity"
    return None, f"invalid_c2_visceral_met_pattern:{vmp}"


def validate_map_result(result, chunks):
    """Validate one chunk map, keeping every finding that passes.

    Returns (normalized_result, fatal_error). Structural problems (not an object,
    missing arrays) are fatal because the response is unusable. Individual bad
    findings are dropped and tallied in the result's `rejected` counter rather
    than discarding the chunk's valid findings alongside them -- at temperature 0
    a rejected chunk would fail identically on every retry, so an all-or-nothing
    validator silently loses the whole chunk permanently.
    """
    if not isinstance(result, dict):
        return None, f"non_dict_response:{type(result).__name__}"
    found = result.get("criteria_found")
    evidence_items = result.get("evidence_items")
    if not isinstance(found, list):
        return None, "missing_criteria_found"
    if not isinstance(evidence_items, list):
        return None, "missing_evidence_items"

    rejected = Counter()
    rejected_items = []

    def record_rejection(reason, item):
        rejected[reason] += 1
        rejected_items.append({"reason": reason, "item": item})

    normalized_found = []
    for item in found:
        normalized, error = _validate_common_finding(item, chunks, date_field="diagnosis_date")
        if error:
            record_rejection(error, item)
            continue
        if not _quote_supports_criterion(
            normalized["criterion"], normalized["quote"]
        ):
            record_rejection(
                f"quote_does_not_support_criterion:{normalized['criterion']}",
                item,
            )
            continue
        vmp, c2_reject = _resolve_c2_pattern(normalized)
        if c2_reject:
            record_rejection(c2_reject, item)
            continue
        normalized["visceral_met_pattern"] = vmp
        normalized_found.append(normalized)

    normalized_evidence = []
    for item in evidence_items:
        # Cap evidence items by truncation rather than rejecting the chunk: the limit
        # exists to bound payload size for the synthesis call, and the prompt already
        # asks the model to prioritize, so keeping the first N loses least.
        if len(normalized_evidence) >= MAX_EVIDENCE_ITEMS_PER_CHUNK:
            record_rejection("evidence_items_over_cap", item)
            continue
        normalized, error = _validate_common_finding(item, chunks, date_field="fact_date")
        if error:
            record_rejection(error, item)
            continue
        fact_type = _normalized_text(item.get("fact_type"))
        fact_value = _normalized_text(item.get("fact_value"))
        if not fact_type or not fact_value:
            record_rejection("missing_fact_type_or_value", item)
            continue
        criterion = normalized["criterion"]
        if fact_type not in VALID_FACT_TYPES_BY_CRITERION[criterion]:
            record_rejection(f"invalid_fact_type:{criterion}:{fact_type}", item)
            continue
        if not _quote_supports_fact_type(fact_type, normalized["quote"]):
            record_rejection(f"quote_does_not_support_fact_type:{fact_type}", item)
            continue
        normalized["candidate_criterion"] = normalized.pop("criterion")
        normalized["fact_type"] = fact_type
        normalized["fact_value"] = fact_value
        normalized_evidence.append(normalized)
    return (
        {
            "criteria_found": normalized_found,
            "evidence_items": normalized_evidence,
            "rejected": dict(rejected),
            "rejected_items": rejected_items,
        },
        None,
    )


def _finding_temporal_key(finding):
    event_date, _, precision = resolve_date(
        finding.get("diagnosis_date"), finding.get("source_note_date")
    )
    precision_rank = {"day": 0, "month": 1, "year": 2, "unknown": 3}
    return (
        event_date or "9999-99-99",
        precision_rank.get(precision, 3),
        finding.get("source_note_date") or "9999-99-99",
    )


def validate_synthesis_result(result, chunks):
    """Validate a patient synthesis, keeping every finding that passes.

    Same partial-success contract as validate_map_result: one hallucinated quote
    among several findings costs that finding, not the patient's entire result.
    """
    if not isinstance(result, dict):
        return None, f"non_dict_response:{type(result).__name__}"
    found = result.get("criteria_found")
    if not isinstance(found, list):
        return None, "missing_criteria_found"
    rejected = Counter()
    rejected_items = []

    def record_rejection(reason, item):
        rejected[reason] += 1
        rejected_items.append({"reason": reason, "item": item})

    by_criterion = {}
    for item in found:
        normalized, error = _validate_common_finding(item, chunks, date_field="diagnosis_date")
        if error:
            record_rejection(error, item)
            continue
        if not _quote_supports_criterion(
            normalized["criterion"], normalized["quote"]
        ):
            record_rejection(
                f"quote_does_not_support_criterion:{normalized['criterion']}",
                item,
            )
            continue
        criterion = normalized["criterion"]
        vmp, c2_reject = _resolve_c2_pattern(normalized)
        if c2_reject:
            record_rejection(c2_reject, item)
            continue
        normalized["visceral_met_pattern"] = vmp
        existing = by_criterion.get(criterion)
        if existing is None:
            by_criterion[criterion] = normalized
            continue
        reason = f"duplicate_criterion:{criterion}"
        if _finding_temporal_key(normalized) < _finding_temporal_key(existing):
            record_rejection(reason, existing)
            by_criterion[criterion] = normalized
        else:
            record_rejection(reason, normalized)
    return {
        "criteria_found": list(by_criterion.values()),
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
        {"role": "system", "content": NEPC_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    result, error = _call_json(provider, client, model, max_retries, messages)
    if error:
        return None, error
    return validate_map_result(result, {chunk_index: chunk})


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
                    "num_criteria": 0,
                    "num_evidence_items": 0,
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
                "num_criteria": len(result["criteria_found"]),
                "num_evidence_items": len(result["evidence_items"]),
                "num_dropped": sum(result.get("rejected", {}).values()),
                "status": (
                    "ok_with_rejections"
                    if result.get("rejected")
                    else "ok"
                ),
                "result_json": json.dumps(result, ensure_ascii=False, sort_keys=True),
            }
        )
    return outputs, chunk_results


def synthesize_patient(provider, client, model, max_retries, mrn, chunk_outputs, chunks):
    payload = {
        "patient_mrn": int(mrn),
        "chunk_maps": [
            # `rejected` is local validation bookkeeping, not evidence -- forwarding it
            # would put our own error taxonomy in the model's context for no benefit.
            {
                "chunk_index": index,
                "criteria_found": result.get("criteria_found", []),
                "evidence_items": result.get("evidence_items", []),
            }
            for index, result in sorted(chunk_outputs.items())
        ],
    }
    messages = [
        {"role": "system", "content": NEPC_SYNTHESIS_PROMPT + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    result, error = _call_json(provider, client, model, max_retries, messages)
    if error:
        return None, error
    return validate_synthesis_result(result, chunks)


def _load_chunk_outputs(path, run_config):
    outputs = {}
    if not path.exists() or path.stat().st_size == 0:
        return outputs
    log = pl.read_csv(path, separator="\t", infer_schema_length=0)
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
    log = pl.read_csv(path, separator="\t", infer_schema_length=0)
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
    log = pl.read_csv(path, separator="\t", infer_schema_length=0)
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
    """Materialize every rejected object for audit/recovery."""
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
        log = pl.read_csv(chunk_log_path, separator="\t", infer_schema_length=0)
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
        log = pl.read_csv(processed_path, separator="\t", infer_schema_length=0)
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


def raw_rows_from_findings(mrn, findings, chunks=None):
    """Convert a validated patient synthesis into raw audit rows."""
    rows = []
    dropped = Counter()
    if isinstance(findings, dict):
        findings = findings.get("criteria_found", [])
    for finding_entry in findings or []:
        legacy_chunk_index = None
        finding = finding_entry
        if isinstance(finding_entry, tuple):
            finding = finding_entry[0]
            if len(finding_entry) >= 3:
                legacy_chunk_index = finding_entry[2]
        criterion = normalize_criterion(finding.get("criterion")) if isinstance(finding, dict) else None
        if criterion is None:
            dropped[str(finding)] += 1
            continue
        chunk_index = finding.get("_support_chunk_index", legacy_chunk_index)
        if chunk_index is None and chunks is not None:
            chunk_index = _find_support_chunk(finding, chunks)
        rows.append(
            {
                "DFCI_MRN": int(mrn),
                "chunk_index": chunk_index,
                "source_note_date": finding.get("source_note_date"),
                "criterion": criterion,
                "diagnosis_date": finding.get("diagnosis_date"),
                "modality": finding.get("modality"),
                "visceral_met_pattern": (
                    "visceral_only" if criterion == "C2" else None
                ),
                "quote": _normalized_text(finding.get("quote")),
                "confidence": finding.get("confidence"),
            }
        )
    return rows, dropped


def rebuild_raw_from_processed(processed_path, raw_path, run_config, patient_chunks):
    completed = _load_completed_syntheses(processed_path, run_config)
    rows = []
    for mrn in sorted(completed):
        if mrn not in patient_chunks:
            continue
        patient_rows, _ = raw_rows_from_findings(
            mrn, completed[mrn], patient_chunks[mrn]
        )
        rows.extend(patient_rows)
    _write_rows_atomic(raw_path, rows, RAW_COLUMNS)
    return len(rows)


def _date_interval(event_date, precision):
    if not event_date:
        return None
    parsed = date.fromisoformat(event_date)
    if precision == "year":
        return date(parsed.year, 1, 1), date(parsed.year, 12, 31)
    if precision == "month":
        last = calendar.monthrange(parsed.year, parsed.month)[1]
        return date(parsed.year, parsed.month, 1), date(parsed.year, parsed.month, last)
    return parsed, parsed


def _prefer_onset(candidate, existing):
    """Prefer certainly earlier dates; within overlapping periods prefer precision."""
    if existing is None:
        return True
    candidate_interval = _date_interval(candidate["event_date"], candidate["date_precision"])
    existing_interval = _date_interval(existing["event_date"], existing["date_precision"])
    if candidate_interval is None:
        return False
    if existing_interval is None:
        return True
    if candidate_interval[1] < existing_interval[0]:
        return True
    if existing_interval[1] < candidate_interval[0]:
        return False
    rank = {"unknown": 0, "year": 1, "month": 2, "day": 3}
    candidate_rank = rank.get(candidate["date_precision"], 0)
    existing_rank = rank.get(existing["date_precision"], 0)
    if candidate_rank != existing_rank:
        return candidate_rank > existing_rank
    return (candidate["source_note_date"] or "9999-99-99") < (
        existing["source_note_date"] or "9999-99-99"
    )


def build_timeline(raw_path, timeline_path):
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        _write_rows_atomic(timeline_path, [], TIMELINE_COLUMNS)
        return 0
    raw = pl.read_csv(raw_path, separator="\t", infer_schema_length=0)
    onsets = {}
    skipped = 0
    for row in raw.iter_rows(named=True):
        criterion = normalize_criterion(row.get("criterion"))
        mrn = _to_exact_int(row.get("DFCI_MRN"))
        if criterion is None or mrn is None:
            skipped += 1
            continue
        event_date, date_source, date_precision = resolve_date(
            row.get("diagnosis_date"), row.get("source_note_date")
        )
        record = {
            "DFCI_MRN": mrn,
            "event_date": event_date,
            "date_source": date_source,
            "date_precision": date_precision,
            "criterion_added": criterion,
            "criterion_label": CRITERION_LABELS[criterion],
            "modality": row.get("modality"),
            "visceral_met_pattern": row.get("visceral_met_pattern"),
            "supporting_quote": row.get("quote"),
            "confidence": row.get("confidence"),
            "source_note_date": row.get("source_note_date"),
        }
        key = (mrn, criterion)
        if _prefer_onset(record, onsets.get(key)):
            onsets[key] = record
    if skipped:
        print(f"  Skipped {skipped} invalid raw rows during timeline build")

    by_patient = {}
    for record in onsets.values():
        by_patient.setdefault(record["DFCI_MRN"], []).append(record)
    output = []
    for mrn in sorted(by_patient):
        events = sorted(
            by_patient[mrn],
            key=lambda item: (item["event_date"] or "9999-99-99", item["criterion_added"]),
        )
        cumulative = set()
        position = 0
        while position < len(events):
            event_date = events[position]["event_date"]
            same_date = []
            while position < len(events) and events[position]["event_date"] == event_date:
                same_date.append(events[position])
                position += 1
            cumulative.update(item["criterion_added"] for item in same_date)
            for item in same_date:
                row = dict(item)
                row["cumulative_criteria"] = " | ".join(sorted(cumulative))
                row["num_criteria_to_date"] = len(cumulative)
                output.append(row)
    _write_rows_atomic(timeline_path, output, TIMELINE_COLUMNS)
    return len(output)


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "avpc_nepc_evidence.tsv")
    meta_path = getattr(args, "evidence_meta_path", None) or _meta_path_for_evidence(
        evidence_path
    )
    raw_path = args.output_dir / "avpc_nepc_extractions_raw.tsv"
    chunk_log_path = args.output_dir / "avpc_nepc_processed_chunks.tsv"
    processed_path = args.output_dir / "avpc_nepc_processed_patients.tsv"
    timeline_path = args.output_dir / "avpc_nepc_timeline.tsv"
    rejected_path = args.output_dir / "avpc_nepc_rejected_findings.tsv"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_nepc_notes.py first."
        )
    if args.overwrite:
        for path in (
            raw_path,
            chunk_log_path,
            processed_path,
            timeline_path,
            rejected_path,
        ):
            path.unlink(missing_ok=True)

    provider = get_provider(args.provider)
    model = args.model or provider.default_model
    scan_config, run_config = check_resume_config(
        chunk_log_path, meta_path, args.provider, model, evidence_path
    )
    print(f"Extraction fingerprint: {run_config} ({args.provider}/{model})")

    evidence_df = pl.read_csv(evidence_path, separator="\t", infer_schema_length=0)
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
                            "num_criteria": 0,
                            "num_evidence_items": 0,
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
                        "num_criteria": 0,
                        "num_dropped": sum(map_rejected.values()),
                        "status": status,
                        "run_config": run_config,
                        "result_json": None,
                    }
                ],
                PROCESSED_COLUMNS,
            )
    print(f"Patient synthesis: {len(synthesis_todo)} outstanding calls")

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
                as_completed(futures), total=len(futures), desc="Synthesize", unit="pt"
            ):
                mrn = futures[future]
                try:
                    _, result, error = future.result()
                except Exception as exc:  # noqa: BLE001
                    result, error = None, f"worker_error:{type(exc).__name__}:{str(exc)[:160]}"
                n_chunks = len(patient_chunks[mrn])
                # Findings the synthesis stage produced but validation refused, plus
                # everything the map stage already dropped for this patient's chunks.
                synthesis_rejected = Counter(result.get("rejected", {})) if result else Counter()
                map_rejected = Counter()
                for chunk_result in chunk_outputs.get(mrn, {}).values():
                    map_rejected.update(chunk_result.get("rejected", {}))
                run_rejected.update(synthesis_rejected)
                append_rows(
                    processed_path,
                    [
                        {
                            "DFCI_MRN": mrn,
                            "num_chunks": n_chunks,
                            "num_chunks_ok": n_chunks,
                            "num_criteria": len(result["criteria_found"]) if result else 0,
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

    raw_count = rebuild_raw_from_processed(
        processed_path, raw_path, run_config, patient_chunks
    )
    rejected_count = rebuild_rejected_findings(
        chunk_log_path, processed_path, rejected_path, run_config
    )
    timeline_count = build_timeline(raw_path, timeline_path)
    print(f"Wrote validated raw findings ({raw_count} rows): {raw_path}")
    print(f"Wrote rejected-finding audit ({rejected_count} rows): {rejected_path}")
    print(f"Wrote AVPC/NEPC timeline ({timeline_count} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
