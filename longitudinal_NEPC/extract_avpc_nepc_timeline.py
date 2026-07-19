"""Pipeline 2 — Longitudinal AVPC / NEPC criteria extraction (per-patient, chunked).

For every prostate patient, notes mentioning AVPC or NEPC language are collected,
de-duplicated, ordered chronologically, and packed into one LLM call per patient
(a few for heavily-documented patients). Each call records which Aparicio
aggressive-variant criteria (C1-C7) and which NEPC sub-features are documented as
present, with the date each was diagnosed. Per-call extractions are aggregated
into a per-patient onset timeline: one row each time a NEW criterion is first added
to the patient's record, carrying the cumulative set of criteria to that date.

Outputs (under <output-dir>):
  avpc_nepc_timeline.tsv          one row per newly-added criterion (with cumulative set)
  avpc_nepc_extractions_raw.tsv   per-finding extractions (provenance, pre-aggregation)
  avpc_nepc_processed_patients.tsv  processed-patient log (resumability + failures)
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.longitudinal_helpers import (  # noqa: E402
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PAYLOAD_MAX_CHARS,
    NEPC_TRIGGER_REGEX,
    PROSTATE_TEXT_CSV,
    build_client,
    call_with_retry,
    filter_note_types,
    flatten_ws,
    group_patient_snippets,
    load_notes,
    load_selected_mrns,
    parse_json_response,
    resolve_date,
)

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_avpc_nepc_timeline"

# Reuse the NEPC classifier's nepc + avpc trigger regexes to collect notes.
TRIGGER_REGEX = {key: NEPC_TRIGGER_REGEX[key] for key in ("nepc", "avpc")}

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

SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved prostate cancer research study.

You will receive a JSON payload with a SINGLE patient's de-identified clinical note
snippets. Each snippet is labeled with its `note_date` and `note_type`, and mentions
language relevant to aggressive-variant prostate cancer (AVPC) or neuroendocrine
prostate cancer (NEPC).

## TASK
Identify which of the following criteria are DOCUMENTED AS PRESENT anywhere in the
snippets. Report each present criterion ONCE, using its EARLIEST documented occurrence,
with that occurrence's date and a verbatim quote.

### Aparicio aggressive-variant criteria (AVPC)
C1 small-cell histology
C2 visceral metastatic pattern — metastasis to lung, adrenal, brain, pleura, or
   peritoneum. Liver / hepatic metastases ALONE do NOT qualify. When C2 is present,
   set visceral_met_pattern: "visceral_only" (no concurrent bone mets) or
   "visceral_and_bone" (with concurrent bone mets).
C3 predominantly lytic bone metastases
C4 bulky disease — restricted to (a) bulky lymphadenopathy / nodal disease, OR
   (b) prostate or pelvic mass with a documented measurement of at least 5 cm.
   Generic "large pelvic mass" / "bulky disease" WITHOUT a >= 5 cm measurement does NOT qualify.
C5 low PSA with high-volume disease
C6 neuroendocrine markers / elevated CEA or LDH / hypercalcemia (when explicit)
C7 rapid progression to castration-resistant or androgen-independent disease

### NEPC sub-features (track each independently)
NEPC:small_cell_dx           neuroendocrine or small-cell prostate carcinoma diagnosis
NEPC:histologic_transformation  histologic transformation from adenocarcinoma to neuroendocrine/small-cell
NEPC:ne_features             neuroendocrine features / differentiation (focal, partial,
                             "with NE features", "component of" all qualify)
NEPC:positive_ne_ihc         positive neuroendocrine IHC on a prostate-derived specimen
                             (synaptophysin, chromogranin, CD56, NSE, INSM1)

## RULES
- Use only the snippets. Report a criterion only when documented as PRESENT — not
  suspected, planned, pending, ruled out, negative, or family history.
- Pathology is most authoritative for histology / IHC; imaging for metastatic pattern.
- diagnosis_date: the date the finding was documented / diagnosed AS STATED in the text
  (YYYY-MM-DD; for partial dates use the first of the month/year). If no date is stated,
  return null.
- source_note_date: the `note_date` of the snippet where the earliest occurrence appears.
  Copy it verbatim from the payload. (Used as a fallback date when diagnosis_date is null.)
- modality: "pathology" | "imaging" | "clinical" | "labs".
- quote: a verbatim excerpt (~30-80 words) supporting the criterion.
- confidence: "high" | "medium" | "low".

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "criteria_found": [
    {"criterion": "C2", "diagnosis_date": "2021-06-01", "source_note_date": "2021-06-03",
     "modality": "imaging", "quote": "<verbatim>", "confidence": "high"}
  ],
  "visceral_met_pattern": "visceral_only | visceral_and_bone | none"
}
If no criteria are documented, return {"criteria_found": [], "visceral_met_pattern": "none"}.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a longitudinal AVPC (C1-C7) / NEPC criteria timeline per prostate patient via the LLM."
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--notes-csv", type=Path, default=PROSTATE_TEXT_CSV)
    parser.add_argument("--note-bundle-path", type=Path, default=None)
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None)
    parser.add_argument(
        "--note-types",
        nargs="+",
        default=None,
        help="Restrict to these NOTE_TYPE values (e.g. Pathology Imaging). Default: all notes.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=2000,
        help="Chars of context kept on each side of an AVPC/NEPC match. Criteria need "
        "broad context (e.g. >= 5 cm measurements, visceral vs bone), so this stays wide.",
    )
    parser.add_argument(
        "--payload-max-chars",
        type=int,
        default=DEFAULT_PAYLOAD_MAX_CHARS,
        help="Max snippet chars packed into one LLM call (one chunk per patient until full).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_rows(path, rows, columns):
    if not rows:
        return
    df = pl.DataFrame({c: [r.get(c) for r in rows] for c in columns})
    write_header = not path.exists() or path.stat().st_size == 0
    text = df.write_csv(separator="\t", include_header=write_header)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def extract_patient(client, model, max_retries, mrn, chunks):
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
            {"role": "system", "content": SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
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
    raw_path = args.output_dir / "avpc_nepc_extractions_raw.tsv"
    processed_path = args.output_dir / "avpc_nepc_processed_patients.tsv"
    timeline_path = args.output_dir / "avpc_nepc_timeline.tsv"

    if args.overwrite:
        for path in (raw_path, processed_path, timeline_path):
            path.unlink(missing_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    notes_df = load_notes(
        csv_path=args.notes_csv,
        bundle_path=args.note_bundle_path,
        raw_text_paths=args.raw_text_path,
        selected_mrns=selected_mrns,
    )
    print(
        f"Loaded notes: {len(notes_df)} rows for "
        f"{notes_df['DFCI_MRN'].n_unique()} patients"
    )

    if args.note_types:
        notes_df = filter_note_types(notes_df, args.note_types)
        print(f"After note-type filter {args.note_types}: {len(notes_df)} rows")

    patient_chunks = group_patient_snippets(
        notes_df,
        TRIGGER_REGEX,
        context_chars=args.context_chars,
        payload_max_chars=args.payload_max_chars,
    )
    total_chunks = sum(len(c) for c in patient_chunks.values())
    print(
        f"Patients mentioning AVPC/NEPC language: {len(patient_chunks)} "
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
        client = build_client()

        def worker(mrn):
            chunks = patient_chunks[mrn]
            findings, error = extract_patient(client, args.model, args.max_retries, mrn, chunks)
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
