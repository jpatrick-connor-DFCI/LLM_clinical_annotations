"""Stage 2 — Call the LLM on per-patient Gleason evidence chunks; write the timeline.

Reads gleason_evidence.tsv produced by collect_gleason_notes.py. Calls the LLM
once per chunk, and aggregates the findings into a deduped per-patient timeline.

Outputs (under <output-dir>):
  gleason_extractions_raw.tsv   per-finding extractions (provenance, pre-dedup)
  gleason_processed_patients.tsv  processed-patient log (resumability + failures)
  gleason_timeline.tsv          deduped timeline (every score + date per patient)

Usage:
  # Run collection first:
  python preprocessing/cli/collect_gleason_notes.py --output-dir /path/to/output

  # Then run LLM extraction:
  python tasks/gleason_score/build_gleason_timeline.py --output-dir /path/to/output
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import CLINICAL_SAFETY_CONTEXT, DEFAULT_DATA_PATH  # noqa: E402
from preprocessing.longitudinal import derive_grade_group, flatten_ws, resolve_date  # noqa: E402
from preprocessing.notes import load_selected_mrns  # noqa: E402
from providers import get_provider  # noqa: E402
from providers.response import parse_json_response  # noqa: E402
from tasks.gleason_score.prompts import GLEASON_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_gleason_timeline"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the LLM on Gleason evidence chunks and write a Gleason timeline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing gleason_evidence.tsv and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help="Override path to gleason_evidence.tsv.")
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
    evidence_path = args.evidence_path or (args.output_dir / "gleason_evidence.tsv")
    raw_path = args.output_dir / "gleason_extractions_raw.tsv"
    processed_path = args.output_dir / "gleason_processed_patients.tsv"
    timeline_path = args.output_dir / "gleason_timeline.tsv"

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n"
            "Run preprocessing/cli/collect_gleason_notes.py first."
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
