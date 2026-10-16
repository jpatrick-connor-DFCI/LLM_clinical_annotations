"""Stage 2 — Per-note AVPC extraction and patient-level AVPC criteria.

Reads avpc_evidence.parquet from preprocessing/cli/collect_avpc_notes.py and
makes ONE LLM call per evidence note. The model reports atomic facts (metastatic
disease, castration resistance, ADT start, small-cell histology, visceral and
bone metastases, bulky disease, Gleason, PSA, NE markers, LDH/CEA,
hypercalcemia), each with a verbatim quote and the date written in the text.
Quotes are grounded against their own note before storage.

The patient rollup (tasks/avpc/criteria.py) is a pure function of the stored
facts: it screens every fact, re-checks every number against its quote, and
applies Aparicio's thresholds, pairings, and timing in code. Criterion numbering
follows the PAPER: C2 = exclusively visceral, C3 = predominantly lytic bone.

Outputs (under <output-dir>):
  avpc_findings.parquet          grounded note-level facts
  avpc_rejected_findings.parquet grounding/validation failures with reasons
  avpc_processed_notes.parquet   per-note ledger, the unit of resume
  avpc_run.parquet               extraction fingerprint
  avpc_labels.parquet            one row per cohort patient: states, C1-C7 with
                                 quote and date, component quotes, is_avpc
  avpc_labels.meta.parquet       criteria and gate versions the labels used

Usage:
  python preprocessing/cli/collect_avpc_notes.py --output-dir /path/to/out
  python tasks/avpc/run_avpc_extraction.py --output-dir /path/to/out --provider vertex_ai
  # Re-apply changed criteria without LLM calls:
  python tasks/avpc/run_avpc_extraction.py --output-dir /path/to/out --rebuild-labels-only
"""

import math
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import AVPC_EVIDENCE_SCHEMA_VERSION, DEFAULT_DATA_PATH  # noqa: E402
from providers import get_provider  # noqa: E402
from tasks import note_extraction  # noqa: E402
from tasks.avpc.criteria import (  # noqa: E402
    COMPONENT_COLUMNS,
    CRITERIA,
    CRITERIA_VERSION,
    evaluate_patient,
)
from tasks.avpc.prompts import (  # noqa: E402
    AVPC_CORRECTION_HINT,
    AVPC_PROMPT_VERSION,
    AVPC_SYSTEM_PROMPT,
    ITEM_FIELDS,
    NE_MARKER_SPECIMENS,
    PSA_CONTEXTS,
)
from tasks.longitudinal_subtype.mcrpc_gate import GATE_VERSION  # noqa: E402
from tasks.longitudinal_subtype.met_veto import VETO_VERSION  # noqa: E402
from tasks.nepc.ne_gate import NE_GATE_VERSION  # noqa: E402

UNLABELED_STATUSES = {"failed", "not_run"}

FINDING_SCHEMA = {
    "item": pl.Utf8,
    "sites": pl.Utf8,
    "agent": pl.Utf8,
    "specimen_site": pl.Utf8,
    "organ": pl.Utf8,
    "exclusively_visceral": pl.Boolean,
    "lytic_predominant": pl.Boolean,
    "count": pl.Int64,
    "high_volume_descriptor": pl.Utf8,
    "size_cm": pl.Float64,
    "gleason_total": pl.Int64,
    "psa_ng_ml": pl.Float64,
    "psa_context": pl.Utf8,
    "marker": pl.Utf8,
    "specimen": pl.Utf8,
    "value": pl.Float64,
    "uln": pl.Float64,
}
TEXT_FIELDS = {"agent", "specimen_site", "organ", "high_volume_descriptor", "marker"}
BOOL_FIELDS = {"exclusively_visceral", "lytic_predominant"}
FLOAT_FIELDS = {"size_cm", "psa_ng_ml", "value", "uln"}
INT_FIELDS = {"count", "gleason_total"}
ENUM_FIELDS = {"psa_context": PSA_CONTEXTS, "specimen": NE_MARKER_SPECIMENS}

LABEL_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "review_status": pl.Utf8,
    "is_metastatic": pl.Boolean,
    "metastatic_quote": pl.Utf8,
    "metastatic_date": pl.Utf8,
    "metastatic_source": pl.Utf8,
    "is_castration_resistant": pl.Boolean,
    "crpc_quote": pl.Utf8,
    "crpc_date": pl.Utf8,
    "crpc_source": pl.Utf8,
    "is_mcrpc": pl.Boolean,
}
for _name in CRITERIA:
    _key = _name.lower()
    LABEL_SCHEMA.update({
        f"{_key}_met": pl.Boolean,
        f"{_key}_quote": pl.Utf8,
        f"{_key}_date": pl.Utf8,
        f"{_key}_source_note_date": pl.Utf8,
    })
LABEL_SCHEMA.update({
    column: (pl.Int64 if column == "c7_interval_days" else pl.Utf8)
    for column in COMPONENT_COLUMNS
})
LABEL_SCHEMA.update({
    "avpc_criteria": pl.List(pl.Utf8),
    "n_avpc_criteria": pl.Int64,
    "is_avpc": pl.Boolean,
    "avpc_date": pl.Utf8,
    "gate_rejections": pl.List(pl.Utf8),
})
BOOLEAN_LABELS = [c for c, t in LABEL_SCHEMA.items() if t == pl.Boolean]


def _as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value or value.lower() in {"null", "none"}:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("not_a_number") from None
    if not math.isfinite(number):
        raise ValueError("not_a_number")
    return number


def _as_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return None
    raise ValueError("not_a_boolean")


def _as_text(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v).strip() for v in value if str(v).strip())
    value = str(value).strip()
    return value or None


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
            if name in FLOAT_FIELDS:
                fields[name] = _as_float(raw)
            elif name in INT_FIELDS:
                number = _as_float(raw)
                if number is not None and not number.is_integer():
                    return None, f"invalid_{name}"
                fields[name] = None if number is None else int(number)
            elif name in BOOL_FIELDS:
                fields[name] = _as_bool(raw)
            elif name in ENUM_FIELDS:
                value = raw.strip().lower() if isinstance(raw, str) else None
                # An unrecognized context is not an error, just not usable.
                fields[name] = value if value in ENUM_FIELDS[name] else None
            else:
                fields[name] = _as_text(raw)
        except ValueError:
            return None, f"invalid_{name}"
    return fields, None


def build_labels(findings, patient_status):
    by_patient = {}
    for row in findings.iter_rows(named=True):
        by_patient.setdefault(int(row["DFCI_MRN"]), []).append(row)
    rows = []
    for mrn, status in patient_status.items():
        fields, rejections = evaluate_patient(by_patient.get(mrn, []))
        if status in UNLABELED_STATUSES:
            # Unread notes are not evidence of absence: leave negatives unset.
            for column in BOOLEAN_LABELS:
                if fields.get(column) is False:
                    fields[column] = None
        rows.append({
            "DFCI_MRN": mrn,
            "review_status": status,
            **fields,
            "gate_rejections": rejections,
        })
    return rows


SPEC = note_extraction.TaskSpec(
    prefix="avpc",
    label="AVPC",
    system_prompt=AVPC_SYSTEM_PROMPT,
    prompt_version=AVPC_PROMPT_VERSION,
    evidence_schema_version=AVPC_EVIDENCE_SCHEMA_VERSION,
    finding_schema=FINDING_SCHEMA,
    validate_finding=validate_finding,
    build_labels=build_labels,
    label_schema=LABEL_SCHEMA,
    label_versions={
        "criteria_version": CRITERIA_VERSION,
        "mcrpc_gate_version": GATE_VERSION,
        "met_veto_version": VETO_VERSION,
        "ne_gate_version": NE_GATE_VERSION,
    },
    correction_hint=AVPC_CORRECTION_HINT,
    default_output_dir=Path(DEFAULT_DATA_PATH) / "LLM_avpc",
    collector_hint="Run preprocessing/cli/collect_avpc_notes.py first.",
)


def run(args):
    # Look the provider up at call time so tests can monkeypatch this module.
    return note_extraction.run(SPEC, args, lambda name: get_provider(name))


def main():
    run(note_extraction.build_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
