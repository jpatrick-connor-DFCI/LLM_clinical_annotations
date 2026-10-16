"""Shared runner for the per-note extraction pipelines (tasks/nepc, tasks/avpc).

Each task makes ONE LLM call per deduplicated trigger-bearing note. The model
returns atomic findings, each with a verbatim quote and a date as written in
the text. Provenance (note_uid, note_date, note_type) comes from the evidence
row, never from the model, so a quote cannot be grounded in one note and dated
from another.

Every finding is grounded against its own note before it is stored. A finding
that fails grounding or validation is written to the rejected-findings file;
the note's other findings are kept. Patient labels use the stored findings
and their source evidence, built by the task's `build_labels`, so
`--rebuild-labels-only` re-applies changed gates or thresholds without LLM calls.

Artifacts (under <output-dir>, see preprocessing.evidence.note_task_paths):
  <prefix>_findings.parquet          grounded note-level findings
  <prefix>_rejected_findings.parquet grounding/validation failures with reasons
  <prefix>_processed_notes.parquet   per-note ledger, the unit of resume
  <prefix>_run.parquet               extraction fingerprint
  <prefix>_labels.parquet            one row per cohort patient
  <prefix>_labels.meta.parquet       gate/criteria versions the labels used

Resume: a note whose ledger row is `ok` or `ok_with_rejections` is never
re-called. A `failed:<err>` note is skipped on a plain re-run and retried with
--retry-failures (which retries only failures), as in run_NEPC_classifier.py.
Every attempt carries an attempt_id; compaction keeps only the findings of each
note's final attempt, so a crash between appends cannot double-count.

The extraction fingerprint covers evidence, provider, model, prompt, and output
schema; any change raises until --overwrite is passed.
"""

import argparse
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import polars as pl
from tqdm.auto import tqdm

from preprocessing.config import CLINICAL_SAFETY_CONTEXT
from preprocessing.evidence import note_task_paths
from preprocessing.grounding import find_quote_support
from preprocessing.longitudinal import (
    file_sha256,
    flatten_ws,
    parse_stated_date,
    read_scan_config_meta,
    resolve_date,
    value_is_missing,
)
from preprocessing.notes import load_selected_mrns, to_iso_date
from preprocessing.parquet_io import read_metadata, write_metadata, write_parquet_atomic
from providers.response import parse_json_response
from tasks.trial_guard import TRIAL_GUARD_VERSION, TrialContext

# Quotes are asked for at <= PROMPT_QUOTE_MAX_CHARS; twice that is accepted so a
# slightly long sentence is not lost, but a pasted paragraph is rejected because
# the gates screen the whole quote and lose specificity on long text.
PROMPT_QUOTE_MAX_CHARS = 300
QUOTE_HARD_MAX_CHARS = 2 * PROMPT_QUOTE_MAX_CHARS
MAX_FINDINGS_PER_NOTE = 40
APPEND_BATCH_NOTES = 200
DEFAULT_OUTPUT_CORRECTION_RETRIES = 1
CONFIDENCE_VALUES = {"high", "medium", "low"}
OK_STATUSES = {"ok", "ok_with_rejections"}

PROVENANCE_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "note_uid": pl.Utf8,
    "source_note_date": pl.Utf8,
    "note_type": pl.Utf8,
    "finding_index": pl.Int64,
}
DATE_QUOTE_SCHEMA = {
    "stated_date": pl.Utf8,
    "event_date": pl.Utf8,
    "date_source": pl.Utf8,
    "date_precision": pl.Utf8,
    "quote": pl.Utf8,
    "confidence": pl.Utf8,
    "attempt_id": pl.Utf8,
}
REJECTED_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "note_uid": pl.Utf8,
    "source_note_date": pl.Utf8,
    "note_type": pl.Utf8,
    "finding_index": pl.Int64,
    "reason": pl.Utf8,
    "finding_json": pl.Utf8,
    "attempt_id": pl.Utf8,
}
LEDGER_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "note_uid": pl.Utf8,
    "note_date": pl.Utf8,
    "note_type": pl.Utf8,
    "num_findings": pl.Int64,
    "num_rejected": pl.Int64,
    "status": pl.Utf8,
    "attempt_id": pl.Utf8,
    "scan_config": pl.Utf8,
    "run_config": pl.Utf8,
}


@dataclass(frozen=True)
class TaskSpec:
    """What a per-note task supplies to the shared runner.

    validate_finding(finding, note) -> (fields, None) | (None, reason)
        Task-specific checks on one model finding AFTER the runner has grounded
        its quote and parsed its stated date. `fields` holds the task columns
        (keys of `finding_schema`).
    build_labels(findings_df, patient_status) -> list[dict]
        Pure rollup from grounded findings to one row per patient in
        `patient_status` ({mrn: review_status}). The runner attaches a transient
        `_trial_context_reason` from source evidence before calling it.
    """

    prefix: str
    label: str
    system_prompt: str
    prompt_version: str
    evidence_schema_version: str
    finding_schema: dict
    validate_finding: Callable
    build_labels: Callable
    label_schema: dict
    label_versions: dict = field(default_factory=dict)
    correction_hint: str = ""
    default_output_dir: Path = None
    collector_hint: str = ""

    @property
    def findings_schema(self):
        return {**PROVENANCE_SCHEMA, **self.finding_schema, **DATE_QUOTE_SCHEMA}


# --- Arguments ----------------------------------------------------------------


def build_parser(spec):
    parser = argparse.ArgumentParser(
        description=f"Run per-note {spec.label} extraction and build patient labels."
    )
    parser.add_argument("--output-dir", type=Path, default=spec.default_output_dir,
                        help=f"Directory containing {spec.prefix}_evidence.parquet "
                             "and where outputs are written.")
    parser.add_argument("--evidence-path", type=Path, default=None,
                        help=f"Override path to {spec.prefix}_evidence.parquet.")
    parser.add_argument("--mrns", default=None,
                        help="Comma-separated MRNs to send to the LLM (pilot runs). "
                             "Labels are still written for the whole cohort.")
    parser.add_argument("--mrn-file", type=Path, default=None,
                        help="CSV cohort file restricting which MRNs are sent to the LLM.")
    parser.add_argument("--provider", choices=["dfci_gpt", "vertex_ai"], default="dfci_gpt",
                        help="Which LLM backend to call.")
    parser.add_argument("--model", default=None,
                        help="Model name override. Defaults to the provider's default_model.")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-notes", type=int, default=None,
                        help="Process at most this many outstanding notes.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard prior extraction outputs and start a new generation.")
    parser.add_argument("--retry-failures", action="store_true",
                        help="Only re-run notes whose ledger status is failed.")
    parser.add_argument("--rebuild-labels-only", action="store_true",
                        help="Rebuild patient labels from stored findings; no LLM calls.")
    return parser


# --- Evidence and fingerprints -------------------------------------------------


def _paths(spec, args):
    paths = note_task_paths(args.output_dir, spec.prefix)
    if getattr(args, "evidence_path", None) is not None:
        evidence = Path(args.evidence_path)
        paths["evidence"] = evidence
        paths["evidence_meta"] = evidence.with_name(f"{evidence.stem}.meta.parquet")
    return paths


def validate_evidence(spec, evidence_path, meta_path):
    """Return the evidence sidecar after checking its version and content hash."""
    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence table not found: {evidence_path}\n{spec.collector_hint}"
        )
    meta = read_scan_config_meta(meta_path)
    if not meta or not meta.get("scan_config"):
        raise ValueError(
            f"Evidence metadata is missing or invalid: {meta_path}. "
            f"Regenerate {spec.label} evidence with --overwrite."
        )
    if meta.get("evidence_schema_version") != spec.evidence_schema_version:
        raise ValueError(
            f"{spec.label} evidence predates the current evidence contract. "
            "Regenerate evidence with --overwrite."
        )
    recorded = meta.get("evidence_sha256")
    if not recorded or recorded != file_sha256(evidence_path):
        raise ValueError(
            f"{spec.label} evidence content does not match its metadata sidecar. "
            "Regenerate evidence with --overwrite."
        )
    return meta


def run_fingerprint(spec, evidence_sha256, provider_name, model):
    """Bind resume state to evidence, backend, prompt, and output contracts."""
    hasher = hashlib.sha256()
    for value in (
        evidence_sha256,
        provider_name,
        model,
        spec.prompt_version,
        spec.system_prompt,
        CLINICAL_SAFETY_CONTEXT,
        correction_message(spec, "<error>"),
        json.dumps({k: str(v) for k, v in spec.findings_schema.items()}),
        json.dumps({k: str(v) for k, v in LEDGER_SCHEMA.items()}),
        json.dumps({k: str(v) for k, v in REJECTED_SCHEMA.items()}),
        str(QUOTE_HARD_MAX_CHARS),
    ):
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:20]


def _validate_run(spec, path, run_config, has_outputs, overwrite):
    if overwrite or not has_outputs:
        write_metadata(path, {"run_config": run_config})
        return
    recorded = (read_metadata(path) or {}).get("run_config")
    if recorded is None:
        raise ValueError(
            f"Existing {spec.label} outputs have no run fingerprint ({path}). "
            "Re-run with --overwrite."
        )
    if recorded != run_config:
        raise ValueError(
            f"{spec.label} provider, model, prompt, schema, or evidence changed "
            f"({recorded} != {run_config}). Re-run with --overwrite to avoid "
            "mixing outputs from two extraction generations."
        )


def _check_ledger_scan_config(ledger_path, scan_config):
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return
    recorded = set(
        pl.read_parquet(ledger_path, columns=["scan_config"])["scan_config"]
        .drop_nulls()
        .to_list()
    )
    if recorded and recorded != {scan_config}:
        raise ValueError(
            "Evidence scan settings differ from the existing note ledger "
            f"({sorted(recorded)} != [{scan_config}]). Restore the matching "
            "evidence or re-run with --overwrite."
        )


def load_evidence_notes(evidence_path):
    """Read evidence into {(mrn, note_uid): note} in a deterministic order."""
    evidence = pl.read_parquet(evidence_path)
    required = {"DFCI_MRN", "note_uid", "note_date", "note_type", "snippet"}
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"Evidence table missing required columns: {missing}")
    notes = {}
    invalid = 0
    for row in evidence.iter_rows(named=True):
        mrn = row.get("DFCI_MRN")
        snippet = row.get("snippet") or ""
        if mrn is None or not row.get("note_uid") or not snippet.strip():
            invalid += 1
            continue
        key = (int(mrn), str(row["note_uid"]))
        if key in notes:
            raise ValueError(f"Duplicate evidence note id: {key}")
        notes[key] = {
            "DFCI_MRN": int(mrn),
            "note_uid": str(row["note_uid"]),
            "note_date": to_iso_date(row.get("note_date")),
            "note_type": row.get("note_type") or "Unknown",
            "snippet": snippet,
        }
    if invalid:
        print(f"  Skipped {invalid} invalid evidence rows")
    return dict(sorted(notes.items(), key=lambda kv: (kv[0][0], kv[1]["note_date"] or "", kv[0][1])))


# --- One note -------------------------------------------------------------------


def correction_message(spec, error):
    return f"""
Your previous answer failed machine validation with: {error}

Re-read the note and return one corrected JSON object only, of the form
{{"findings": [ ... ]}}. Use {{"findings": []}} when nothing qualifies.
- Every quote must be a short, exact, contiguous substring of note_text
  (at most {PROMPT_QUOTE_MAX_CHARS} characters). Never paraphrase or splice passages.
- stated_date is the date written in the text at its stated precision
  (YYYY, YYYY-MM, or YYYY-MM-DD), or null. Never copy note_date into it.
- Stay concise: report each distinct finding once.
{spec.correction_hint}
""".strip()


def _call_note(spec, provider, client, model, max_retries, note, correction_retries):
    """Return (findings_list, error). Retries once on parse/truncation errors."""
    payload = {
        "note_type": note["note_type"],
        "note_date": note["note_date"],
        "note_text": note["snippet"],
    }
    messages = [
        {"role": "system", "content": spec.system_prompt + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    last_error = None
    for attempt in range(correction_retries + 1):
        attempt_messages = messages
        if last_error is not None:
            attempt_messages = messages + [
                {"role": "user", "content": correction_message(spec, last_error)}
            ]
        text, error = provider.call_with_retry(client, model, attempt_messages, max_retries)
        if error:
            last_error = error
            # Transport retries are exhausted inside the adapter; only a
            # truncated answer benefits from asking again for a shorter one.
            if not str(error).startswith("truncated_response"):
                return None, error
            continue
        try:
            result = parse_json_response(text)
        except json.JSONDecodeError as exc:
            last_error = f"json_parse: {exc}"
            continue
        if not isinstance(result, dict):
            last_error = f"non_dict_response: {type(result).__name__}"
            continue
        findings = result.get("findings")
        if not isinstance(findings, list):
            last_error = "missing_findings_list"
            continue
        return findings, None
    return None, last_error or "no_result"


def ground_finding(spec, finding, note):
    """Ground one model finding in its note; return (row_fields, None) or (None, reason)."""
    if not isinstance(finding, dict):
        return None, "finding_not_object"
    quote = finding.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        return None, "missing_quote"
    flat_quote = flatten_ws(quote)
    if len(flat_quote) > QUOTE_HARD_MAX_CHARS:
        return None, "quote_too_long"
    support = find_quote_support(
        quote, [{"snippet": note["snippet"], "note_date": note["note_date"]}]
    )
    if support is None:
        return None, "quote_not_in_note"

    stated = finding.get("stated_date")
    if value_is_missing(stated) or str(stated).strip().lower() in {"", "null", "none", "unknown"}:
        stated = None
    if stated is not None:
        stated_iso, _ = parse_stated_date(stated)
        if stated_iso is None:
            return None, f"invalid_stated_date:{stated}"
        if note["note_date"] and stated_iso > note["note_date"]:
            return None, "stated_date_after_note"
    event_date, date_source, date_precision = resolve_date(stated, note["note_date"])

    fields, reason = spec.validate_finding(finding, note)
    if reason:
        return None, reason
    confidence = finding.get("confidence")
    confidence = confidence.lower() if isinstance(confidence, str) else None
    return {
        **fields,
        "stated_date": str(stated) if stated is not None else None,
        "event_date": event_date,
        "date_source": date_source,
        "date_precision": date_precision,
        "quote": flat_quote,
        "confidence": confidence if confidence in CONFIDENCE_VALUES else None,
    }, None


def extract_note(spec, provider, client, model, max_retries, note,
                 correction_retries=DEFAULT_OUTPUT_CORRECTION_RETRIES):
    """Run one note end to end; return (finding_rows, rejected_rows, ledger_row)."""
    attempt_id = uuid.uuid4().hex
    provenance = {
        "DFCI_MRN": note["DFCI_MRN"],
        "note_uid": note["note_uid"],
        "source_note_date": note["note_date"],
        "note_type": note["note_type"],
    }
    findings, error = _call_note(
        spec, provider, client, model, max_retries, note, correction_retries
    )
    ledger = {
        "DFCI_MRN": note["DFCI_MRN"],
        "note_uid": note["note_uid"],
        "note_date": note["note_date"],
        "note_type": note["note_type"],
        "num_findings": 0,
        "num_rejected": 0,
        "attempt_id": attempt_id,
    }
    if error:
        return [], [], {**ledger, "status": f"failed:{error}"}

    rows, rejected = [], []
    for index, finding in enumerate(findings):
        if index >= MAX_FINDINGS_PER_NOTE:
            reason, fields = "too_many_findings", None
        else:
            fields, reason = ground_finding(spec, finding, note)
        if reason:
            rejected.append({
                **provenance,
                "finding_index": index,
                "reason": reason,
                "finding_json": json.dumps(finding, ensure_ascii=False, default=str),
                "attempt_id": attempt_id,
            })
            continue
        rows.append({**provenance, "finding_index": index, **fields, "attempt_id": attempt_id})
    status = "ok_with_rejections" if rejected else "ok"
    return rows, rejected, {
        **ledger,
        "num_findings": len(rows),
        "num_rejected": len(rejected),
        "status": status,
    }


# --- Persistence ------------------------------------------------------------------


def _frame(rows, schema):
    return pl.DataFrame(
        {c: [row.get(c) for row in rows] for c in schema},
        schema=schema,
        strict=False,
    )


def _read(path, schema):
    if not path.exists() or path.stat().st_size == 0:
        return _frame([], schema)
    frame = pl.read_parquet(path)
    missing = [c for c in schema if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; re-run with --overwrite.")
    return frame.select([pl.col(c).cast(t, strict=False) for c, t in schema.items()])


def _append(path, rows, schema):
    if not rows:
        return
    incoming = _frame(rows, schema)
    if path.exists() and path.stat().st_size:
        incoming = pl.concat([_read(path, schema), incoming], how="vertical")
    write_parquet_atomic(incoming, path)


def compact_outputs(paths, spec):
    """Keep the last ledger row per note and only that attempt's findings."""
    ledger = _read(paths["processed"], LEDGER_SCHEMA)
    if ledger.is_empty():
        return ledger
    before = ledger.height
    ledger = ledger.unique(subset=["DFCI_MRN", "note_uid"], keep="last", maintain_order=True)
    if ledger.height != before:
        write_parquet_atomic(ledger, paths["processed"])
        print(f"  Compacted {paths['processed'].name}: {before} -> {ledger.height} rows")
    final = ledger.select(["DFCI_MRN", "note_uid", "attempt_id"])
    for key, schema in (("findings", spec.findings_schema), ("rejected", REJECTED_SCHEMA)):
        frame = _read(paths[key], schema)
        kept = frame.join(final, on=["DFCI_MRN", "note_uid", "attempt_id"], how="semi")
        if kept.height != frame.height:
            write_parquet_atomic(kept, paths[key])
            print(f"  Compacted {paths[key].name}: {frame.height} -> {kept.height} rows")
    return ledger


# --- Labels -----------------------------------------------------------------------


def patient_review_status(notes, ledger, cohort_mrns):
    """{mrn: status} from the ledger: ok / partial:K/N / failed / not_run / no_trigger_notes.

    K counts notes without a usable result (failed or never attempted) out of
    the patient's N evidence notes.
    """
    by_note = {
        (int(r["DFCI_MRN"]), r["note_uid"]): r["status"]
        for r in ledger.iter_rows(named=True)
    }
    per_patient = {}
    for (mrn, uid) in notes:
        per_patient.setdefault(mrn, []).append(by_note.get((mrn, uid)))
    status = {}
    for mrn, statuses in per_patient.items():
        total = len(statuses)
        ok = sum(1 for s in statuses if s in OK_STATUSES)
        attempted = sum(1 for s in statuses if s is not None)
        if ok == total:
            status[mrn] = "ok"
        elif attempted == 0:
            status[mrn] = "not_run"
        elif ok == 0 and attempted == total:
            status[mrn] = "failed"
        else:
            status[mrn] = f"partial:{total - ok}/{total}"
    for mrn in cohort_mrns:
        status.setdefault(int(mrn), "no_trigger_notes")
    return dict(sorted(status.items()))


def annotate_trial_context(findings, notes):
    """Attach source-derived gate reasons without changing stored extraction rows.

    This runs on both extraction and label-only rebuilds, so existing grounded
    findings can be screened using their original evidence without new calls.
    """
    contexts = {}
    reasons = []
    for row in findings.iter_rows(named=True):
        key = (int(row["DFCI_MRN"]), row["note_uid"])
        if key not in contexts:
            note = notes.get(key)
            contexts[key] = TrialContext(note["snippet"]) if note else None
        context = contexts[key]
        reasons.append(context.reason(row["quote"]) if context else None)
    return findings.with_columns(pl.Series("_trial_context_reason", reasons, dtype=pl.Utf8))


def write_labels(spec, paths, findings, status, run_config, notes=None):
    if notes is not None:
        findings = annotate_trial_context(findings, notes)
    rows = spec.build_labels(findings, status)
    labels = _frame(rows, spec.label_schema)
    write_parquet_atomic(labels, paths["labels"])
    write_metadata(
        paths["labels_meta"],
        {"run_config": run_config, "trial_guard_version": TRIAL_GUARD_VERSION,
         **{k: str(v) for k, v in spec.label_versions.items()}},
    )
    return labels


# --- Entry point ------------------------------------------------------------------


def run(spec, args, get_provider):
    """Execute one task. `get_provider` is injected so tests can fake the backend."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(spec, args)
    outputs = [paths[k] for k in ("findings", "rejected", "processed", "labels", "labels_meta")]
    rebuild_only = getattr(args, "rebuild_labels_only", False)

    if args.overwrite and rebuild_only:
        raise ValueError("--overwrite and --rebuild-labels-only are mutually exclusive.")
    if args.overwrite:
        for path in (*outputs, paths["run"]):
            path.unlink(missing_ok=True)

    meta = validate_evidence(spec, paths["evidence"], paths["evidence_meta"])
    scan_config = meta["scan_config"]
    _check_ledger_scan_config(paths["processed"], scan_config)
    notes = load_evidence_notes(paths["evidence"])
    evidence_mrns = {mrn for mrn, _ in notes}
    cohort_mrns = set(meta.get("cohort_mrns") or []) | evidence_mrns
    print(
        f"Loaded {spec.label} evidence: {len(notes)} notes for {len(evidence_mrns)} "
        f"patients (cohort {len(cohort_mrns)})"
    )

    if rebuild_only:
        run_config = (read_metadata(paths["run"]) or {}).get("run_config")
        if run_config is None:
            raise ValueError(
                f"No {spec.label} extraction run found in {args.output_dir}; "
                "run the extraction before --rebuild-labels-only."
            )
        print(f"Rebuilding labels from stored findings (run {run_config}); no LLM calls.")
    else:
        provider = get_provider(args.provider)
        model = args.model or provider.default_model
        run_config = run_fingerprint(spec, meta["evidence_sha256"], args.provider, model)
        _validate_run(
            spec, paths["run"], run_config,
            any(p.exists() for p in outputs[:3]), args.overwrite,
        )
        print(f"Extraction fingerprint: {run_config} ({args.provider}/{model})")

        ledger = compact_outputs(paths, spec)
        prior = {
            (int(r["DFCI_MRN"]), r["note_uid"]): r["status"]
            for r in ledger.iter_rows(named=True)
        }
        selected = load_selected_mrns(args.mrns, getattr(args, "mrn_file", None))
        if args.retry_failures:
            todo = [
                note for key, note in notes.items()
                if prior.get(key, "").startswith("failed")
            ]
            print(f"Retrying {len(todo)} failed notes")
        else:
            todo = [note for key, note in notes.items() if key not in prior]
        if selected is not None:
            todo = [note for note in todo if note["DFCI_MRN"] in selected]
        if args.limit_notes is not None:
            todo = todo[: args.limit_notes]
        print(f"Notes to extract with LLM: {len(todo)} (already in ledger: {len(prior)})")

        if todo:
            client = provider.build_client()
            buffers = {"findings": [], "rejected": [], "processed": []}

            def flush():
                # Ledger last: a crash before it leaves the note unlogged, so it
                # is re-run and compaction drops the orphaned attempt's findings.
                _append(paths["findings"], buffers["findings"], spec.findings_schema)
                _append(paths["rejected"], buffers["rejected"], REJECTED_SCHEMA)
                _append(paths["processed"], buffers["processed"], LEDGER_SCHEMA)
                for rows in buffers.values():
                    rows.clear()

            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                futures = [
                    executor.submit(
                        extract_note, spec, provider, client, model, args.max_retries, note
                    )
                    for note in todo
                ]
                pending = 0
                try:
                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc="Notes", unit="note"
                    ):
                        rows, rejected, ledger_row = future.result()
                        buffers["findings"].extend(rows)
                        buffers["rejected"].extend(rejected)
                        buffers["processed"].append(
                            {**ledger_row, "scan_config": scan_config, "run_config": run_config}
                        )
                        pending += 1
                        if pending >= APPEND_BATCH_NOTES:
                            flush()
                            pending = 0
                finally:
                    flush()

    ledger = compact_outputs(paths, spec)
    findings = _read(paths["findings"], spec.findings_schema)
    status = patient_review_status(notes, ledger, cohort_mrns)
    counts = {}
    for value in status.values():
        bucket = value.split(":", 1)[0]
        counts[bucket] = counts.get(bucket, 0) + 1
    labels = write_labels(spec, paths, findings, status, run_config, notes)
    print(f"Review status: {dict(sorted(counts.items()))}")
    print(f"Wrote {spec.label} labels ({labels.height} patients): {paths['labels']}")
    return labels
