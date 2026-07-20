import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.llm_helpers import (  # noqa: E402
    CLASSIFY_SYSTEM_PROMPT,
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    PROSTATE_TEXT_CSV,
    build_client,
    build_patient_snippets,
    call_with_retry,
    load_notes,
    load_selected_mrns,
    parse_json_response,
    resolve_note_source,
)


OUTPUT_COLUMNS = [
    "DFCI_MRN",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify each prostate patient as NEPC / AVPC / biomarker / conventional with one LLM call."
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument(
        "--notes-csv",
        type=Path,
        default=PROSTATE_TEXT_CSV,
        help="Compiled prostate notes CSV (default note source).",
    )
    parser.add_argument(
        "--note-bundle-path",
        type=Path,
        default=None,
        help="Optional gzipped note bundle. Overrides the CSV when it exists; "
        "falls back to raw OncDRS JSONs if neither is present.",
    )
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--max-notes-per-patient", type=int, default=75)
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=None,
        help="Processes for the note clean/trigger/snippet scan (default: all cores).",
    )
    parser.add_argument(
        "--no-snippet-cache",
        action="store_true",
        help="Disable the per-cohort snippet cache (forces a full re-scan).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _append_tsv_row(path, row, columns):
    """Append a single row to a TSV, writing the header only on first write.

    Polars has no append mode for write_csv, so the header/row text is written
    directly with a file handle kept open in append mode.
    """
    write_header = not path.exists() or path.stat().st_size == 0
    df = pl.DataFrame({c: [row.get(c)] for c in columns})
    text = df.write_csv(separator="\t", include_header=write_header)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def append_row(path, row):
    _append_tsv_row(path, row, OUTPUT_COLUMNS)


def append_failure(path, mrn, error, num_snippets):
    row = {"DFCI_MRN": int(mrn), "error": error, "num_snippets": int(num_snippets)}
    _append_tsv_row(path, row, list(row.keys()))


def classify_patient(client, model, max_retries, mrn, snippets):
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
    response_text, error = call_with_retry(client, model, messages, max_retries)
    if error:
        return None, error
    try:
        result = parse_json_response(response_text)
    except json.JSONDecodeError as exc:
        return None, f"json_parse: {exc}"
    if not isinstance(result, dict):
        return None, f"non_dict_response: {type(result).__name__}"
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


def make_row(mrn, num_snippets, result):
    return {
        "DFCI_MRN": int(mrn),
        "primary_label": result.get("primary_label"),
        "has_nepc": result.get("has_nepc"),
        "has_avpc": result.get("has_avpc"),
        "has_biomarker": result.get("has_biomarker"),
        "has_molecular_avpc": result.get("has_molecular_avpc"),
        "has_non_prostate_primary": result.get("has_non_prostate_primary"),
        "biomarker_genes": " | ".join(str(g) for g in _as_list(result.get("biomarker_genes"))),
        "avpc_criteria": " | ".join(str(c) for c in _as_list(result.get("avpc_criteria"))),
        "visceral_met_pattern": result.get("visceral_met_pattern"),
        "non_prostate_primary_types": " | ".join(
            str(t) for t in _as_list(result.get("non_prostate_primary_types"))
        ),
        "supporting_quotes": " | ".join(str(q) for q in _as_list(result.get("supporting_quotes"))),
        "supporting_quote_dates": " | ".join(str(d) for d in _as_list(result.get("supporting_quote_dates"))),
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale"),
        "num_snippets": int(num_snippets),
    }


def conventional_row(mrn):
    return {
        "DFCI_MRN": int(mrn),
        "primary_label": "conventional",
        "has_nepc": False,
        "has_avpc": False,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": "",
        "avpc_criteria": "",
        "visceral_met_pattern": "none",
        "non_prostate_primary_types": "",
        "supporting_quotes": "",
        "supporting_quote_dates": "",
        "confidence": "high",
        "rationale": "No NEPC / AVPC / biomarker / non-prostate-primary triggers found in any reviewed note.",
        "num_snippets": 0,
    }


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "LLM_NEPC_classifier_labels.tsv"
    failures_path = args.output_dir / "LLM_NEPC_classifier_failed_patients.tsv"

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    bundle_path = args.note_bundle_path

    source_label, source_path = resolve_note_source(
        csv_path=args.notes_csv, bundle_path=bundle_path
    )
    if source_label == "raw_json":
        print(
            "Note source: raw OncDRS JSONs (no bundle or CSV found) — this is the "
            "slowest path; build the CSV/bundle to speed up loading."
        )
    else:
        print(f"Note source: {source_label} ({source_path})")

    notes_df = load_notes(
        csv_path=args.notes_csv,
        bundle_path=bundle_path,
        raw_text_paths=args.raw_text_path,
        selected_mrns=selected_mrns,
    )
    print(
        f"Loaded notes: {len(notes_df)} rows for "
        f"{notes_df['DFCI_MRN'].n_unique()} patients"
    )

    cache_dir = None if args.no_snippet_cache else (args.output_dir / "snippet_cache")
    patient_snippets = build_patient_snippets(
        notes_df,
        max_notes_per_patient=args.max_notes_per_patient,
        max_workers=args.scan_workers,
        cache_dir=cache_dir,
    )
    all_mrns = set(notes_df["DFCI_MRN"].cast(pl.Int64).unique().to_list())
    triggered_mrns = set(patient_snippets.keys())
    no_signal_mrns = all_mrns - triggered_mrns

    print(f"Patients with triggered snippets: {len(triggered_mrns)}")
    print(f"Patients with no signal (auto-conventional): {len(no_signal_mrns)}")

    completed = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        completed = set(
            pl.read_csv(output_path, separator="\t")["DFCI_MRN"].cast(pl.Int64).to_list()
        )
    print(f"Already completed: {len(completed)}")

    mrns_to_run = sorted(triggered_mrns - completed)
    if args.limit_mrns is not None:
        mrns_to_run = mrns_to_run[: args.limit_mrns]
    print(f"Patients to classify with LLM: {len(mrns_to_run)}")

    no_signal_to_write = sorted(no_signal_mrns - completed)

    if not mrns_to_run:
        for mrn in no_signal_to_write:
            append_row(output_path, conventional_row(mrn))
        print(f"Wrote labels: {output_path}")
        return

    client = build_client()

    def worker(mrn):
        snippets = patient_snippets[mrn]
        result, error = classify_patient(client, args.model, args.max_retries, mrn, snippets)
        return mrn, snippets, result, error

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all LLM work first so calls are in flight immediately, then write
        # the no-signal (auto-conventional) rows while the API calls run — the
        # no-signal write no longer delays time-to-first-call.
        futures = {executor.submit(worker, mrn): mrn for mrn in mrns_to_run}
        for mrn in no_signal_to_write:
            append_row(output_path, conventional_row(mrn))
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Patients", unit="pt"
        ):
            mrn, snippets, result, error = future.result()
            if error or result is None:
                tqdm.write(f"  Classification failed for {mrn}: {error}")
                append_failure(failures_path, mrn, error or "no_result", len(snippets))
                continue
            append_row(output_path, make_row(mrn, len(snippets), result))

    print(f"Wrote labels: {output_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
