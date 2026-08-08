import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.bundles.snippet_bundle import (  # noqa: E402
    SNIPPET_BUNDLE_FILENAME,
    load_snippet_bundle,
)
from preprocessing.config import CLINICAL_SAFETY_CONTEXT, DEFAULT_OUTPUT_DIR  # noqa: E402
from preprocessing.grounding import find_quote_support  # noqa: E402
from preprocessing.notes import load_selected_mrns  # noqa: E402
from preprocessing.longitudinal import file_sha256  # noqa: E402
from preprocessing.parquet_io import (  # noqa: E402
    append_rows_atomic,
    read_metadata,
    write_metadata,
    write_parquet_atomic,
)
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.binary_NEPC.prompts import CLASSIFY_SYSTEM_PROMPT  # noqa: E402


OUTPUT_COLUMNS = [
    "DFCI_MRN",
    "review_status",
    "primary_label",
    "has_nepc",
    "has_avpc",
    "has_biomarker",
    "has_molecular_avpc",
    "has_non_prostate_primary",
    "biomarker_genes",
    "avpc_criteria",
    "visceral_met_pattern",
    "non_prostate_primary_types",
    "supporting_quotes",
    "supporting_quote_dates",
    "confidence",
    "rationale",
    "num_snippets",
]
FAILURE_COLUMNS = ["DFCI_MRN", "error", "num_snippets"]
_PRIMARY_LABELS = {"nepc", "avpc", "biomarker", "conventional"}
_CONFIDENCE_LEVELS = {"high", "medium", "low"}
_AVPC_CRITERIA = {f"C{index}" for index in range(1, 8)}
BINARY_EXTRACTION_SCHEMA_VERSION = "binary-nepc-grounded-parquet-v4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify each prostate patient as NEPC / AVPC / biomarker / conventional with one LLM call."
    )
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="CSV cohort file containing DFCI_MRN values.")
    parser.add_argument("--mrns", default=None)
    parser.add_argument(
        "--snippets-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / SNIPPET_BUNDLE_FILENAME,
        help=(
            "Patient snippet bundle produced by preprocessing/cli/compile_patient_snippets.py. "
            "The classifier never reads or scans raw notes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-mrns", type=int, default=None)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--overwrite", action="store_true")
    run_mode.add_argument(
        "--retry-failures",
        action="store_true",
        help="Only rerun MRNs currently listed in the failed-patients Parquet.",
    )
    return parser.parse_args()


def append_row(path, row):
    append_rows_atomic(path, [row], OUTPUT_COLUMNS)


def read_mrns(path):
    """Read patient identifiers from an existing pipeline Parquet artifact."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pl.read_parquet(path)
    if "DFCI_MRN" not in frame.columns:
        raise ValueError(f"Missing DFCI_MRN column in {path}")
    return set(
        frame["DFCI_MRN"].cast(pl.Int64, strict=False).drop_nulls().to_list()
    )


def remove_failures(path, mrns):
    """Remove resolved patients from the failure Parquet artifact."""
    mrns = {int(mrn) for mrn in mrns}
    if not mrns or not path.exists() or path.stat().st_size == 0:
        return
    frame = pl.read_parquet(path)
    if "DFCI_MRN" not in frame.columns:
        raise ValueError(f"Missing DFCI_MRN column in {path}")
    remaining = frame.filter(
        ~pl.col("DFCI_MRN").cast(pl.Int64, strict=False).is_in(sorted(mrns))
    )
    if remaining.height == frame.height:
        return
    write_parquet_atomic(remaining, path)


def append_failure(path, mrn, error, num_snippets):
    row = {"DFCI_MRN": int(mrn), "error": error, "num_snippets": int(num_snippets)}
    # Keep only the latest error when the same patient fails repeated retries.
    remove_failures(path, [mrn])
    append_rows_atomic(path, [row], FAILURE_COLUMNS)


def classify_patient(provider, client, model, max_retries, mrn, snippets):
    payload = {
        "patient_mrn": int(mrn),
        "notes": [
            {
                "note_date": s["note_date"],
                "note_type": s["note_type"],
                "trigger_categories": s["trigger_categories"],
                "note_text": s["snippet"],
            }
            for s in snippets
        ],
    }
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
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
    return validate_result(result, snippets)


def validate_result(result, snippets):
    """Validate the classifier schema, cross-field invariants, and quote provenance."""
    boolean_fields = (
        "has_nepc",
        "has_avpc",
        "has_biomarker",
        "has_molecular_avpc",
        "has_non_prostate_primary",
    )
    for field in boolean_fields:
        if not isinstance(result.get(field), bool):
            return None, f"invalid_boolean:{field}"

    primary_label = result.get("primary_label")
    if primary_label not in _PRIMARY_LABELS:
        return None, f"invalid_primary_label:{primary_label}"
    expected_label = (
        "nepc"
        if result["has_nepc"]
        else "avpc"
        if result["has_avpc"]
        else "biomarker"
        if result["has_biomarker"]
        else "conventional"
    )
    if primary_label != expected_label:
        return None, f"inconsistent_primary_label:{primary_label}!={expected_label}"

    list_fields = (
        "biomarker_genes",
        "avpc_criteria",
        "non_prostate_primary_types",
        "supporting_quotes",
        "supporting_quote_dates",
    )
    for field in list_fields:
        if not isinstance(result.get(field), list):
            return None, f"invalid_list:{field}"

    criteria = result["avpc_criteria"]
    if any(value not in _AVPC_CRITERIA for value in criteria):
        return None, "invalid_avpc_criterion"
    if criteria and not result["has_avpc"]:
        return None, "avpc_criteria_without_has_avpc"
    visceral_pattern = result.get("visceral_met_pattern")
    if visceral_pattern not in {"visceral_only", "none"}:
        return None, f"invalid_visceral_met_pattern:{visceral_pattern}"
    if ("C2" in criteria) != (visceral_pattern == "visceral_only"):
        return None, "inconsistent_c2_visceral_pattern"

    normalized_genes = {str(value).strip().upper() for value in result["biomarker_genes"]}
    expected_biomarker = bool(normalized_genes & {"BRCA1", "BRCA2"})
    if result["has_biomarker"] != expected_biomarker:
        return None, "inconsistent_has_biomarker"
    expected_molecular_avpc = len(normalized_genes & {"PTEN", "TP53", "RB1"}) >= 2
    if result["has_molecular_avpc"] != expected_molecular_avpc:
        return None, "inconsistent_has_molecular_avpc"
    if result["has_non_prostate_primary"] != bool(result["non_prostate_primary_types"]):
        return None, "inconsistent_non_prostate_primary"

    confidence = result.get("confidence")
    if confidence not in _CONFIDENCE_LEVELS:
        return None, f"invalid_confidence:{confidence}"
    if not isinstance(result.get("rationale"), str) or not result["rationale"].strip():
        return None, "missing_rationale"

    quotes = result["supporting_quotes"]
    quote_dates = result["supporting_quote_dates"]
    if len(quotes) != len(quote_dates):
        return None, "quote_date_count_mismatch"
    if any(result[field] for field in boolean_fields) and not quotes:
        return None, "positive_result_without_supporting_quote"
    for quote, quote_date in zip(quotes, quote_dates):
        if not isinstance(quote, str) or not quote.strip():
            return None, "invalid_supporting_quote"
        support = find_quote_support(quote, snippets, claimed_date=quote_date)
        if support is None or support.get("note_date") != quote_date:
            return None, "supporting_quote_or_date_not_in_evidence"

    return result, None


def _as_list(value):
    """Coerce an LLM field to a list. A bare string is wrapped (not iterated as
    characters); None/empty becomes []. Guards against the model returning e.g.
    biomarker_genes="BRCA2" instead of ["BRCA2"], which would otherwise serialize
    as "B | R | C | A | 2"."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_string_set(value):
    """Return a deterministic, case-insensitively deduplicated string set."""
    unique = {}
    for item in _as_list(value):
        text = str(item).strip()
        if text:
            unique.setdefault(text.casefold(), text)
    return [unique[key] for key in sorted(unique)]


def make_row(mrn, num_snippets, result):
    return {
        "DFCI_MRN": int(mrn),
        "review_status": "llm_classified",
        "primary_label": result.get("primary_label"),
        "has_nepc": result.get("has_nepc"),
        "has_avpc": result.get("has_avpc"),
        "has_biomarker": result.get("has_biomarker"),
        "has_molecular_avpc": result.get("has_molecular_avpc"),
        "has_non_prostate_primary": result.get("has_non_prostate_primary"),
        "biomarker_genes": _as_string_set(result.get("biomarker_genes")),
        "avpc_criteria": _as_string_set(result.get("avpc_criteria")),
        "visceral_met_pattern": result.get("visceral_met_pattern"),
        "non_prostate_primary_types": _as_string_set(
            result.get("non_prostate_primary_types")
        ),
        "supporting_quotes": [str(q) for q in _as_list(result.get("supporting_quotes"))],
        "supporting_quote_dates": [
            str(d) for d in _as_list(result.get("supporting_quote_dates"))
        ],
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale"),
        "num_snippets": int(num_snippets),
    }


def conventional_row(mrn):
    return {
        "DFCI_MRN": int(mrn),
        "review_status": "no_trigger",
        "primary_label": "conventional",
        "has_nepc": False,
        "has_avpc": False,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": [],
        "avpc_criteria": [],
        "visceral_met_pattern": "none",
        "non_prostate_primary_types": [],
        "supporting_quotes": [],
        "supporting_quote_dates": [],
        "confidence": "high",
        "rationale": "No NEPC / AVPC / biomarker / non-prostate-primary triggers found in any reviewed note.",
        "num_snippets": 0,
    }


def no_notes_row(mrn):
    row = {column: None for column in OUTPUT_COLUMNS}
    row.update({
        "DFCI_MRN": int(mrn),
        "review_status": "no_notes",
        "rationale": "No PROFILE_DATA clinical notes were available for this cohort patient.",
        "num_snippets": 0,
    })
    return row


def _run_fingerprint(snippets_path, provider_name, model):
    hasher = hashlib.sha256()
    for value in (
        file_sha256(snippets_path),
        provider_name,
        model,
        CLASSIFY_SYSTEM_PROMPT,
        CLINICAL_SAFETY_CONTEXT,
        BINARY_EXTRACTION_SCHEMA_VERSION,
        json.dumps(OUTPUT_COLUMNS),
    ):
        hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()[:20]


def _validate_run_fingerprint(path, run_config, has_existing_outputs, overwrite):
    if overwrite or not has_existing_outputs:
        write_metadata(path, {"run_config": run_config})
        return
    if not path.exists():
        raise ValueError(
            "Existing binary NEPC outputs predate run fingerprinting. Re-run with "
            "--overwrite rather than mixing them with the current snippet bundle."
        )
    try:
        recorded = (read_metadata(path) or {}).get("run_config")
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Invalid binary NEPC run metadata: {path}") from exc
    if recorded != run_config:
        raise ValueError(
            f"Binary NEPC inputs/config changed ({recorded} != {run_config}). "
            "Re-run with --overwrite."
        )


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "LLM_NEPC_classifier_labels.parquet"
    failures_path = args.output_dir / "LLM_NEPC_classifier_failed_patients.parquet"
    run_meta_path = args.output_dir / "LLM_NEPC_classifier_run.parquet"

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)

    provider = get_provider(args.provider)
    model = args.model or provider.default_model
    run_config = _run_fingerprint(args.snippets_path, args.provider, model)
    _validate_run_fingerprint(
        run_meta_path,
        run_config,
        output_path.exists() or failures_path.exists(),
        args.overwrite,
    )

    completed = read_mrns(output_path)
    failed = read_mrns(failures_path)
    # Repair stale state left by older runs, which did not remove a successful
    # retry from the failure/unlabeled list.
    remove_failures(failures_path, completed)
    failed -= completed

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    if getattr(args, "retry_failures", False):
        if selected_mrns is not None:
            failed &= selected_mrns
        selected_mrns = failed
        if not selected_mrns:
            print(f"No failed patients to retry: {failures_path}")
            return
        print(f"Retrying failed patients: {len(selected_mrns)}")

    all_mrns, patient_snippets, snippet_metadata = load_snippet_bundle(
        args.snippets_path,
        selected_mrns=selected_mrns,
    )
    if selected_mrns is not None and not all_mrns:
        raise ValueError(
            f"None of the selected MRNs are present in {args.snippets_path}"
        )
    print(f"Loaded patient snippets: {args.snippets_path}")
    print(f"Snippet cohort patients: {len(all_mrns)}")
    if snippet_metadata:
        print(f"Snippet compilation metadata: {json.dumps(snippet_metadata)}")

    triggered_mrns = set(patient_snippets.keys())
    no_note_mrns = {
        int(mrn) for mrn in snippet_metadata.get("no_note_mrns", [])
    } & all_mrns
    no_signal_mrns = all_mrns - triggered_mrns - no_note_mrns

    print(f"Patients with triggered snippets: {len(triggered_mrns)}")
    print(f"Patients with no signal (auto-conventional): {len(no_signal_mrns)}")
    print(f"Patients with no notes (unclassified): {len(no_note_mrns)}")

    print(f"Already completed: {len(completed)}")

    mrns_to_run = sorted(triggered_mrns - completed)
    if args.limit_mrns is not None:
        mrns_to_run = mrns_to_run[: args.limit_mrns]
    print(f"Patients to classify with LLM: {len(mrns_to_run)}")

    no_signal_to_write = sorted(no_signal_mrns - completed)
    no_notes_to_write = sorted(no_note_mrns - completed)

    if not mrns_to_run:
        for mrn in no_signal_to_write:
            append_row(output_path, conventional_row(mrn))
        for mrn in no_notes_to_write:
            append_row(output_path, no_notes_row(mrn))
        remove_failures(failures_path, no_signal_to_write + no_notes_to_write)
        print(f"Wrote labels: {output_path}")
        return

    client = provider.build_client()

    def worker(mrn):
        snippets = patient_snippets[mrn]
        result, error = classify_patient(provider, client, model, args.max_retries, mrn, snippets)
        return mrn, snippets, result, error

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all LLM work first so calls are in flight immediately, then write
        # the no-signal (auto-conventional) rows while the API calls run — the
        # no-signal write no longer delays time-to-first-call.
        futures = {executor.submit(worker, mrn): mrn for mrn in mrns_to_run}
        for mrn in no_signal_to_write:
            append_row(output_path, conventional_row(mrn))
        for mrn in no_notes_to_write:
            append_row(output_path, no_notes_row(mrn))
        remove_failures(failures_path, no_signal_to_write + no_notes_to_write)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Patients", unit="pt"
        ):
            mrn, snippets, result, error = future.result()
            if error or result is None:
                tqdm.write(f"  Classification failed for {mrn}: {error}")
                append_failure(failures_path, mrn, error or "no_result", len(snippets))
                continue
            append_row(output_path, make_row(mrn, len(snippets), result))
            remove_failures(failures_path, [mrn])

    print(f"Wrote labels: {output_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
