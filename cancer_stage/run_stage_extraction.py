"""Stage 2 — Call the LLM on per-patient evidence chunks; write the stage timeline.

Reads stage_evidence.tsv produced by extract_stage_notes.py. Groups snippets by
patient into payload-sized chunks (chronological, greedy packing), calls the LLM
once per chunk, and aggregates the findings into a deduped stage timeline.

Outputs (under <output-dir>):
  stage_extractions_raw.tsv      Per-finding extractions (one row per LLM finding,
                                 pre-dedup, with rationale for auditing).
  stage_processed_patients.tsv   Per-patient processing log (resumability + failures).
  stage_timeline.tsv             Deduped stage timeline — one row per distinct staging
                                 event per patient.

Usage:
  # Run scan first:
  python cancer_stage/extract_stage_notes.py --output-dir /path/to/output

  # Then run LLM extraction:
  python cancer_stage/run_stage_extraction.py --output-dir /path/to/output
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.longitudinal_helpers import (  # noqa: E402
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_MODEL_NAME,
    DEFAULT_PAYLOAD_MAX_CHARS,
    build_client,
    call_with_retry,
    flatten_ws,
    load_selected_mrns,
    parse_json_response,
    resolve_date,
)

from cancer_stage.prompts import STAGE_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("STAGE_OUTPUT_DIR", "/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/")
)

RAW_COLUMNS = [
    "DFCI_MRN",
    "source_note_date",
    "cancer_type",
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
    "stage_group",
    "stage_date",
    "date_source",
    "is_historical_reference",
    "supporting_quote",
    "confidence",
    "source_note_date",
]

PROCESSED_COLUMNS = ["DFCI_MRN", "num_chunks", "num_findings", "status"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the LLM on stage evidence chunks and write a stage timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing stage_evidence.tsv and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help="Override path to stage_evidence.tsv.")
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="Process only these MRNs (file of MRNs or CSV with DFCI_MRN column).")
    parser.add_argument("--mrns", default=None,
                        help="Comma- or space-separated MRNs to process.")
    parser.add_argument("--payload-max-chars", type=int, default=DEFAULT_PAYLOAD_MAX_CHARS,
                        help="Max snippet chars packed into one LLM call (one chunk per patient "
                             "until the budget is full).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
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
    for row in evidence_df.itertuples(index=False):
        mrn = int(row.DFCI_MRN)
        by_mrn.setdefault(mrn, []).append({
            "note_date": row.note_date if pd.notna(row.note_date) else None,
            "note_type": row.note_type if pd.notna(row.note_type) else "Unknown",
            "trigger_categories": (
                str(row.trigger_categories).split(",")
                if pd.notna(row.trigger_categories) and row.trigger_categories
                else []
            ),
            "snippet": row.snippet if pd.notna(row.snippet) else "",
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


def extract_patient(client, model, max_retries, mrn, chunks):
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
        response_text, error = call_with_retry(client, model, messages, max_retries)
        if error:
            if error.startswith("content_filter"):
                # Content filter errors are deterministic — retrying the same chunk
                # will never succeed. Skip this chunk and preserve findings from
                # other chunks rather than failing the whole patient permanently.
                continue
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
        findings.extend(f for f in chunk_findings if isinstance(f, dict))
    return findings, None


def _normalize_stage_group(val):
    """Strip leading 'Stage ' prefix and uppercase — e.g. 'Stage IV' → 'IV', 'IIIa' → 'IIIA'."""
    if not val:
        return None
    cleaned = re.sub(r"(?i)^stage\s+", "", str(val).strip())
    return cleaned.upper() or None


def raw_rows_from_findings(mrn, findings):
    rows = []
    for finding in findings:
        rows.append({
            "DFCI_MRN": int(mrn),
            "source_note_date": finding.get("source_note_date"),
            "cancer_type": finding.get("cancer_type"),
            "stage_group": _normalize_stage_group(finding.get("stage_group")),
            "stage_date": finding.get("stage_date"),
            "is_historical_reference": finding.get("is_historical_reference"),
            "supporting_quote": flatten_ws(finding.get("supporting_quote")),
            "confidence": finding.get("confidence"),
            "rationale": flatten_ws(finding.get("rationale")),
        })
    return rows


def append_rows(path, rows, columns):
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        mode="a",
        sep="\t",
        index=False,
        header=not path.exists() or path.stat().st_size == 0,
    )


def _str(val):
    """Return val as a stripped string, treating None and NaN as empty string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def build_timeline(raw_path, timeline_path):
    """Deduplicate raw findings into the stage timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pd.DataFrame(columns=TIMELINE_COLUMNS).to_csv(timeline_path, sep="\t", index=False)
        return 0

    raw = pd.read_csv(raw_path, sep="\t", dtype=str, on_bad_lines="skip")
    seen = set()
    rows = []
    for r in raw.itertuples(index=False):
        mrn_val = pd.to_numeric(getattr(r, "DFCI_MRN", None), errors="coerce")
        if pd.isna(mrn_val):
            continue
        mrn = int(mrn_val)

        # Normalize dedup key fields so formatting differences don't create duplicates.
        cancer_type_raw = _str(getattr(r, "cancer_type", None))
        stage_group_raw = _str(getattr(r, "stage_group", None))

        stage_date, date_source = resolve_date(
            getattr(r, "stage_date", None), getattr(r, "source_note_date", None)
        )

        key = (
            mrn,
            cancer_type_raw.lower() or None,
            stage_group_raw.upper() or None,
            stage_date,
        )
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "DFCI_MRN": mrn,
            "cancer_type": cancer_type_raw or None,
            "stage_group": stage_group_raw or None,
            "stage_date": stage_date,
            "date_source": date_source,
            "is_historical_reference": getattr(r, "is_historical_reference", None),
            "supporting_quote": getattr(r, "supporting_quote", None),
            "confidence": getattr(r, "confidence", None),
            "source_note_date": getattr(r, "source_note_date", None),
        })

    timeline = pd.DataFrame(rows, columns=TIMELINE_COLUMNS)
    if not timeline.empty:
        timeline = timeline.sort_values(
            ["DFCI_MRN", "cancer_type", "stage_date"], na_position="last"
        )
        # Keep only rows where stage_group changes within each (patient, cancer_type).
        # This collapses repeated identical staging entries over time — once a stage
        # is established (including metastatic/IV), subsequent rows with the same
        # stage add no new information.
        last_stage = {}
        keep = []
        for idx, row in timeline.iterrows():
            key = (row["DFCI_MRN"], (_str(row["cancer_type"])).lower())
            curr = (_str(row["stage_group"])).upper()
            if last_stage.get(key) != curr:
                keep.append(idx)
                last_stage[key] = curr
        timeline = timeline.loc[keep]
    timeline.to_csv(timeline_path, sep="\t", index=False)
    return len(timeline)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_path or (args.output_dir / "stage_evidence.tsv")
    raw_path = args.output_dir / "stage_extractions_raw.tsv"
    processed_path = args.output_dir / "stage_processed_patients.tsv"
    timeline_path = args.output_dir / "stage_timeline.tsv"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run extract_stage_notes.py first."
        )

    if args.overwrite:
        for path in (raw_path, processed_path, timeline_path):
            path.unlink(missing_ok=True)

    evidence_df = pd.read_csv(evidence_path, sep="\t", dtype=str)
    evidence_df["DFCI_MRN"] = pd.to_numeric(evidence_df["DFCI_MRN"], errors="coerce")
    evidence_df = evidence_df.dropna(subset=["DFCI_MRN"])
    evidence_df["DFCI_MRN"] = evidence_df["DFCI_MRN"].astype(int)
    print(
        f"Loaded evidence: {len(evidence_df)} snippets for "
        f"{evidence_df['DFCI_MRN'].nunique()} patients"
    )

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    if selected_mrns is not None:
        evidence_df = evidence_df.loc[evidence_df["DFCI_MRN"].isin(selected_mrns)].copy()
        print(f"After MRN filter: {len(evidence_df)} snippets for "
              f"{evidence_df['DFCI_MRN'].nunique()} patients")

    patient_chunks = group_evidence_chunks(evidence_df, args.payload_max_chars)
    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients to process: {len(patient_chunks)} "
        f"({total_chunks} LLM calls across chunks)"
    )

    completed = set()
    if processed_path.exists() and processed_path.stat().st_size > 0:
        log = pd.read_csv(processed_path, sep="\t")
        completed = set(log.loc[log["status"] == "ok", "DFCI_MRN"].astype(int))
    print(f"Already completed patients: {len(completed)}")

    todo = [m for m in sorted(patient_chunks) if m not in completed]
    if args.limit_patients is not None:
        todo = todo[: args.limit_patients]
    print(f"Patients to extract with LLM: {len(todo)}")

    if todo:
        client = build_client()

        def worker(mrn):
            chunks = patient_chunks[mrn]
            findings, error = extract_patient(
                client, args.model, args.max_retries, mrn, chunks
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
