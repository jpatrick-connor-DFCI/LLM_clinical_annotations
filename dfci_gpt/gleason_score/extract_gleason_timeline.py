"""Pipeline 1 — Longitudinal Gleason score extraction (per-patient, chunked).

For every prostate patient, notes mentioning a Gleason score / Grade Group / ISUP
grade are collected, de-duplicated, ordered chronologically, and packed into one
LLM call per patient (a few for heavily-documented patients). Each call extracts
every documented Gleason score with the date the grade was assigned. The results
are aggregated and de-duplicated into a per-patient timeline: every distinct
Gleason score the patient received, with its date.

Outputs (under <output-dir>):
  gleason_timeline.tsv          deduped timeline (every score + date per patient)
  gleason_extractions_raw.tsv   per-finding extractions (provenance, pre-dedup)
  gleason_processed_patients.tsv  processed-patient log (resumability + failures)
"""

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

from shared.longitudinal_helpers import (  # noqa: E402
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PAYLOAD_MAX_CHARS,
    PROSTATE_TEXT_CSV,
    build_client,
    call_with_retry,
    derive_grade_group,
    filter_note_types,
    flatten_ws,
    group_patient_snippets,
    load_notes,
    load_selected_mrns,
    parse_json_response,
    resolve_date,
)

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_gleason_timeline"

# Any mention of Gleason / Grade Group / ISUP grading collects the note.
TRIGGER_REGEX = {
    "gleason": r"\b(?:gleason|grade\s+group|isup(?:\s+grade)?)\b",
}

RAW_COLUMNS = [
    "DFCI_MRN",
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

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "gleason_date",
    "date_source",
    "gleason_primary",
    "gleason_secondary",
    "gleason_total",
    "grade_group",
    "specimen_type",
    "is_historical_reference",
    "supporting_quote",
    "source_note_date",
]

PROCESSED_COLUMNS = ["DFCI_MRN", "num_chunks", "num_findings", "status"]

SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved prostate cancer research study.

You will receive a JSON payload with a SINGLE patient's de-identified clinical note
snippets. Each snippet is labeled with its `note_date` and `note_type`, and was
selected because it mentions a Gleason score, Grade Group, or ISUP grade.

## TASK
Extract EVERY distinct Gleason score documented ACROSS ALL of the snippets. The same
score is often restated in many notes (copy-forward); report each distinct score once.
For each distinct score, report:
- primary: primary Gleason pattern as an integer 1-5 (null if only a grade group is given).
- secondary: secondary Gleason pattern as an integer 1-5 (null if only a grade group is given).
- total: total Gleason sum as an integer 2-10 (null if not derivable from the text).
- grade_group: ISUP Grade Group 1-5 if explicitly stated; otherwise null (it will be derived).
- specimen_type: one of "biopsy", "prostatectomy", "TURP", "metastasis", "unknown".
- scoring_date: the date the specimen was obtained / the grade was originally assigned,
  AS STATED in the text (YYYY-MM-DD; for partial dates use the first of the month/year).
  If no date is stated for this score, return null.
- source_note_date: the `note_date` of the snippet where you found this score. Copy it
  verbatim from the payload. (Used as a fallback date when scoring_date is null.)
- is_historical_reference: true if the score is quoted from a prior/outside report;
  false if it is the result being newly reported in that note.
- quote: a verbatim excerpt (~20-60 words) containing the score.

## RULES
- Extract only scores explicitly documented. Never infer or compute a score that is not written.
- Treat separate specimens or separate dates as separate entries; do not merge them.
- If the identical score (same patterns/total) is documented for the same specimen/date in
  several notes, report it once, using the EARLIEST note_date as source_note_date.
- Planned, pending, or "awaiting" pathology is NOT a score.

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "gleason_findings": [
    {"primary": 4, "secondary": 3, "total": 7, "grade_group": 3,
     "specimen_type": "biopsy", "scoring_date": "2019-03-01",
     "source_note_date": "2019-03-05", "is_historical_reference": false,
     "quote": "<verbatim>"}
  ]
}
If no actual Gleason score is documented, return {"gleason_findings": []}.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a longitudinal Gleason-score timeline per prostate patient via the LLM."
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
        help="Restrict to these NOTE_TYPE values (e.g. Pathology). Default: all notes. "
        "Gleason is authoritatively assigned in pathology, so 'Pathology' is far "
        "cheaper and higher-fidelity.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=600,
        help="Chars of context kept on each side of a Gleason match. Smaller windows "
        "raise the copy-forward dedup hit-rate and pack more notes per call.",
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


def extract_patient(client, model, max_retries, mrn, chunks):
    """Run one LLM call per chunk; return the merged findings list for the patient."""
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
        chunk_findings = result.get("gleason_findings")
        if not isinstance(chunk_findings, list):
            return None, "missing_gleason_findings"
        findings.extend(f for f in chunk_findings if isinstance(f, dict))
    return findings, None


def raw_rows_from_findings(mrn, findings):
    rows = []
    for finding in findings:
        rows.append({
            "DFCI_MRN": int(mrn),
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
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_timeline(raw_path, timeline_path):
    """Resolve dates, validate, and de-duplicate raw extractions into the timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pl.DataFrame(schema={c: pl.Utf8 for c in TIMELINE_COLUMNS}).write_csv(
            timeline_path, separator="\t"
        )
        return 0

    # Read every field as text and validate per row, so a single malformed/misaligned
    # row (e.g. free-text that shifted columns) can't abort the whole timeline build.
    raw = pl.read_csv(raw_path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)
    seen = set()
    rows = []
    skipped = 0
    for r in raw.iter_rows(named=True):
        mrn_val = _to_int(r.get("DFCI_MRN"))
        if mrn_val is None:
            skipped += 1
            continue
        mrn = mrn_val

        primary = _to_int(r.get("gleason_primary"))
        secondary = _to_int(r.get("gleason_secondary"))
        total = _to_int(r.get("gleason_total"))
        # Gleason total is primary + secondary by definition; recompute it when
        # both patterns are present so an LLM arithmetic slip can't propagate.
        if primary is not None and secondary is not None:
            total = primary + secondary
        # Require a usable total; drop grade-group-only or malformed extractions.
        if total is None or not (2 <= total <= 10):
            continue
        if primary is not None and not (1 <= primary <= 5):
            continue
        if secondary is not None and not (1 <= secondary <= 5):
            continue

        grade_group = _to_int(r.get("grade_group"))
        if grade_group is None or not (1 <= grade_group <= 5):
            grade_group = derive_grade_group(primary, secondary)

        gleason_date, date_source = resolve_date(
            r.get("scoring_date"), r.get("source_note_date")
        )
        specimen_type = r.get("specimen_type")

        key = (mrn, primary, secondary, total, gleason_date, specimen_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "DFCI_MRN": mrn,
            "gleason_date": gleason_date,
            "date_source": date_source,
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
    timeline.write_csv(timeline_path, separator="\t")
    return timeline.height


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "gleason_extractions_raw.tsv"
    processed_path = args.output_dir / "gleason_processed_patients.tsv"
    timeline_path = args.output_dir / "gleason_timeline.tsv"

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
        f"Patients mentioning Gleason: {len(patient_chunks)} "
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
                        [{"DFCI_MRN": int(mrn), "num_chunks": n_chunks, "num_findings": 0,
                          "status": error or "no_result"}],
                        PROCESSED_COLUMNS,
                    )
                    continue
                rows = raw_rows_from_findings(mrn, findings)
                append_rows(raw_path, rows, RAW_COLUMNS)
                append_rows(
                    processed_path,
                    [{"DFCI_MRN": int(mrn), "num_chunks": n_chunks, "num_findings": len(rows),
                      "status": "ok"}],
                    PROCESSED_COLUMNS,
                )

    n = build_timeline(raw_path, timeline_path)
    print(f"Wrote Gleason timeline ({n} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
