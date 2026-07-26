"""Stage 2 — Call the LLM on per-patient AVPC/NEPC evidence chunks; write the timeline.

Reads avpc_nepc_evidence.tsv produced by collect_nepc_notes.py. Calls the LLM once
per chunk, recording which Aparicio aggressive-variant criteria (C1-C7) and which
NEPC sub-features are documented as present, with the date each was diagnosed.
Per-call extractions are aggregated into a per-patient onset timeline: one row each
time a NEW criterion is first added to the patient's record, carrying the cumulative
set of criteria to that date.

Outputs (under <output-dir>):
  avpc_nepc_extractions_raw.tsv   per-finding extractions (provenance, pre-aggregation)
  avpc_nepc_processed_patients.tsv  processed-patient log (resumability + failures)
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

PROCESSED_COLUMNS = ["DFCI_MRN", "num_chunks", "num_criteria", "status"]


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


def extract_patient(provider, client, model, max_retries, mrn, chunks):
    """Run one LLM call per chunk; return the merged criteria findings for the patient."""
    findings = []
    for chunk in chunks:
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
        for finding in found:
            if isinstance(finding, dict):
                findings.append((finding, vmp))
    return findings, None


def raw_rows_from_findings(mrn, findings):
    rows = []
    for finding, vmp in findings:
        criterion = finding.get("criterion")
        if criterion not in VALID_CRITERIA:
            continue
        rows.append({
            "DFCI_MRN": int(mrn),
            "source_note_date": finding.get("source_note_date"),
            "criterion": criterion,
            "diagnosis_date": finding.get("diagnosis_date"),
            "modality": finding.get("modality"),
            "visceral_met_pattern": vmp if criterion == "C2" else None,
            "quote": flatten_ws(finding.get("quote")),
            "confidence": finding.get("confidence"),
        })
    return rows


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
    processed_path = args.output_dir / "avpc_nepc_processed_patients.tsv"
    timeline_path = args.output_dir / "avpc_nepc_timeline.tsv"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_nepc_notes.py first."
        )

    if args.overwrite:
        for path in (raw_path, processed_path, timeline_path):
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
    for row in evidence_df.iter_rows(named=True):
        mrn = int(row["DFCI_MRN"])
        rec = {
            "note_date": row.get("note_date"),
            "note_type": row.get("note_type") or "Unknown",
            "snippet": row.get("snippet") or "",
        }
        chunk_index = int(row.get("chunk_index") or 0)
        chunks = patient_chunks.setdefault(mrn, [])
        while len(chunks) <= chunk_index:
            chunks.append([])
        chunks[chunk_index].append(rec)

    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients to process: {len(patient_chunks)} "
        f"({total_chunks} LLM calls across chunks)"
    )

    completed = set()
    if processed_path.exists() and processed_path.stat().st_size > 0:
        log = pl.read_csv(processed_path, separator="\t")
        completed = set(
            log.filter(pl.col("status") == "ok")["DFCI_MRN"].cast(pl.Int64).to_list()
        )
    print(f"Already completed patients: {len(completed)}")

    todo = [m for m in sorted(patient_chunks) if m not in completed]
    if args.limit_patients is not None:
        todo = todo[: args.limit_patients]
    print(f"Patients to extract with LLM: {len(todo)}")

    if todo:
        provider = get_provider(args.provider)
        model = args.model or provider.default_model
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
                        [{"DFCI_MRN": int(mrn), "num_chunks": n_chunks, "num_criteria": 0,
                          "status": error or "no_result"}],
                        PROCESSED_COLUMNS,
                    )
                    continue
                rows = raw_rows_from_findings(mrn, findings)
                append_rows(raw_path, rows, RAW_COLUMNS)
                append_rows(
                    processed_path,
                    [{"DFCI_MRN": int(mrn), "num_chunks": n_chunks, "num_criteria": len(rows),
                      "status": "ok"}],
                    PROCESSED_COLUMNS,
                )

    n = build_timeline(raw_path, timeline_path)
    print(f"Wrote AVPC/NEPC criteria timeline ({n} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
