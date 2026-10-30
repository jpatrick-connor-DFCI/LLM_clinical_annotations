"""Stage 2 — Per-note disease-course extraction (shared by BCR, progression, mCRPC).

Reads course_evidence.parquet from preprocessing/cli/collect_course_notes.py and
makes ONE LLM call per evidence note. The model reports atomic dated facts: PSA
values, prostatectomy, radiation, systemic therapy start/current/stop,
recurrence and progression statements, imaging recurrence, and CRPC / mCRPC
statements, each with a verbatim quote and the date written in the text.

Extraction only checks structure: every number must appear in its quote, a PSA
needs a stated date or "current" timing, a therapy start needs a stated date.
Whether a fact was planned, negated, hedged or trial text is decided at label
time by tasks/disease_course/statement_gate.py, so a gate change is re-applied
with --rebuild-labels-only and by every builder without LLM calls.

Outputs (under <output-dir>):
  course_findings.parquet          grounded note-level facts
  course_rejected_findings.parquet grounding/validation failures with reasons
  course_processed_notes.parquet   per-note ledger, the unit of resume
  course_run.parquet               extraction fingerprint
  course_labels.parquet            one row per cohort patient: RP / primary RT /
                                   salvage RT dates, local therapy, first note
                                   ADT start, PSA count, gate rejections
  course_labels.meta.parquet       gate versions the labels used

The endpoint labels are built from these findings by:
  tasks/bcr/build_bcr_labels.py
  tasks/progression/build_progression_events.py
  tasks/mcrpc/build_mcrpc_labels.py

Usage:
  python preprocessing/cli/collect_course_notes.py --output-dir /path/to/out
  python tasks/disease_course/run_course_extraction.py --output-dir /path/to/out --provider vertex_ai
  python tasks/disease_course/run_course_extraction.py --output-dir /path/to/out --rebuild-labels-only
"""

import re
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import COURSE_EVIDENCE_SCHEMA_VERSION, DEFAULT_DATA_PATH  # noqa: E402
from providers import get_provider  # noqa: E402
from tasks import note_extraction  # noqa: E402
from tasks.avpc.criteria import number_in_quote  # noqa: E402
from tasks.avpc.run_avpc_extraction import _as_float, _as_text  # noqa: E402
from tasks.disease_course.course import local_therapy  # noqa: E402
from tasks.disease_course.prompts import (  # noqa: E402
    COURSE_CORRECTION_HINT,
    COURSE_PROMPT_VERSION,
    COURSE_SYSTEM_PROMPT,
    ENUMS,
    ITEM_FIELDS,
)
from tasks.disease_course.statement_gate import (  # noqa: E402
    COURSE_GATE_VERSION,
    usable_course_findings,
)
from tasks.disease_course.therapy import THERAPY_VERSION  # noqa: E402
from tasks.longitudinal_subtype.mcrpc_gate import GATE_VERSION  # noqa: E402
from tasks.longitudinal_subtype.met_veto import VETO_VERSION  # noqa: E402
from tasks.nepc.ne_gate import NE_GATE_VERSION  # noqa: E402

UNLABELED_STATUSES = {"failed", "not_run"}
DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_disease_course"
MAX_FINDINGS_PER_NOTE = 120
PSA_PLAUSIBLE = (0.0, 10_000.0)

FINDING_SCHEMA = {"item": pl.Utf8}
for _fields in ITEM_FIELDS.values():
    for _name in _fields:
        FINDING_SCHEMA[_name] = pl.Float64 if _name == "psa_ng_ml" else pl.Utf8

UNDETECTABLE_CUE = re.compile(
    r"\b(?:undetect\w*|not\s+detect\w*|non[\s-]?detect\w*|below\s+(?:the\s+)?"
    r"(?:limit|detection|lln)|<\s*lln)",
    flags=re.IGNORECASE,
)
_BOUND_BEFORE = re.compile(r"(<=|>=|≤|≥|<|>|less\s+than|greater\s+than)\s*$", re.IGNORECASE)
_LESS = {"<", "<=", "≤", "less than"}

LABEL_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "review_status": pl.Utf8,
    "local_therapy": pl.Utf8,
}
for _prefix in ("rp", "primary_rt", "salvage_rt", "first_note_adt"):
    LABEL_SCHEMA.update({
        f"{_prefix}_date": pl.Utf8,
        f"{_prefix}_date_precision": pl.Utf8,
        f"{_prefix}_date_source": pl.Utf8,
        f"{_prefix}_quote": pl.Utf8,
    })
LABEL_SCHEMA.update({
    "salvage_rt_intent": pl.Utf8,
    "n_psa_values": pl.Int64,
    "gate_rejections": pl.List(pl.Utf8),
})


def _bound_written(value, quote):
    """The comparator written right before `value` in the quote, if any."""
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)(?![\d.])", quote or ""):
        if abs(float(match.group(1)) - value) < 1e-9:
            bound = _BOUND_BEFORE.search(quote[: match.start()])
            if bound:
                text = re.sub(r"\s+", " ", bound.group(1).lower())
                return "<" if text in _LESS else ">"
    return None


def _validate_psa(fields, quote):
    value = fields["psa_ng_ml"]
    comparator = fields["value_comparator"]
    if value is not None:
        if not PSA_PLAUSIBLE[0] <= value <= PSA_PLAUSIBLE[1]:
            return "implausible_psa"
        if not number_in_quote(value, quote):
            return "psa_not_in_quote"
        written = _bound_written(value, quote)
        if written is not None:
            comparator = written
        elif comparator in (None, "undetectable"):
            # "undetectable (<0.05)" keeps the number as a bound.
            comparator = "<" if comparator == "undetectable" else "="
    else:
        if comparator == "undetectable" or UNDETECTABLE_CUE.search(quote):
            if UNDETECTABLE_CUE.search(quote) is None:
                return "undetectable_without_cue"
            comparator = "undetectable"
        else:
            return "psa_without_value"
    fields["value_comparator"] = comparator
    return None


def validate_finding(finding, note):
    item = finding.get("item")
    item = item.strip().lower() if isinstance(item, str) else item
    if item not in ITEM_FIELDS:
        return None, f"invalid_item:{item}"
    fields = {column: None for column in FINDING_SCHEMA}
    fields["item"] = item
    for name in ITEM_FIELDS[item]:
        raw = finding.get(name)
        try:
            if name == "psa_ng_ml":
                fields[name] = _as_float(raw)
            elif name in ENUMS:
                value = raw.strip().lower() if isinstance(raw, str) else None
                fields[name] = value if value in ENUMS[name] else None
            else:
                fields[name] = _as_text(raw)
        except ValueError:
            return None, f"invalid_{name}"
    quote = finding.get("quote") or ""
    stated = finding.get("stated_date")
    has_stated = isinstance(stated, str) and stated.strip().lower() not in {"", "null", "none", "unknown"}
    if item == "psa_value":
        reason = _validate_psa(fields, quote)
        if reason:
            return None, reason
        if has_stated:
            fields["value_timing"] = "dated"
        elif fields["value_timing"] != "current":
            return None, "psa_undated"
    if item in {"therapy_start", "therapy_current", "therapy_stop"} and not fields["agent"]:
        return None, "missing_agent"
    if item == "therapy_start" and not has_stated:
        return None, "therapy_start_undated"
    return fields, None


def build_labels(findings, patient_status):
    usable, rejected = usable_course_findings(findings)
    rows = []
    for mrn, status in patient_status.items():
        patient = usable.get(mrn, [])
        facts = local_therapy(patient)
        facts = {k: v for k, v in facts.items() if not k.startswith("_")}
        n_psa = sum(1 for r in patient if r["item"] == "psa_value")
        if status in UNLABELED_STATUSES:
            # Unread notes are not evidence: report nothing for this patient.
            facts = {k: None for k in facts}
            n_psa = None
        rows.append({
            "DFCI_MRN": mrn,
            "review_status": status,
            **facts,
            "n_psa_values": n_psa,
            "gate_rejections": rejected.get(mrn, []),
        })
    return rows


SPEC = note_extraction.TaskSpec(
    prefix="course",
    label="disease course",
    system_prompt=COURSE_SYSTEM_PROMPT,
    prompt_version=COURSE_PROMPT_VERSION,
    evidence_schema_version=COURSE_EVIDENCE_SCHEMA_VERSION,
    finding_schema=FINDING_SCHEMA,
    validate_finding=validate_finding,
    build_labels=build_labels,
    label_schema=LABEL_SCHEMA,
    label_versions={
        "course_gate_version": COURSE_GATE_VERSION,
        "therapy_version": THERAPY_VERSION,
        "mcrpc_gate_version": GATE_VERSION,
        "met_veto_version": VETO_VERSION,
        "ne_gate_version": NE_GATE_VERSION,
    },
    correction_hint=COURSE_CORRECTION_HINT,
    default_output_dir=DEFAULT_OUTPUT_DIR,
    collector_hint="Run preprocessing/cli/collect_course_notes.py first.",
    max_findings_per_note=MAX_FINDINGS_PER_NOTE,
)


def run(args):
    # Look the provider up at call time so tests can monkeypatch this module.
    return note_extraction.run(SPEC, args, lambda name: get_provider(name))


def main():
    run(note_extraction.build_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
