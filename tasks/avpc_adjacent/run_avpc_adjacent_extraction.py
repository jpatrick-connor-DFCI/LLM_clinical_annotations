"""Stage 2 — Per-note AVPC-adjacent extraction and patient-level labels.

Reads avpc_adjacent_evidence.parquet from
preprocessing/cli/collect_avpc_adjacent_notes.py and makes ONE LLM call per
evidence note, with the same shared runner as tasks/avpc. The model reports the
AVPC atomic facts (with an extended visceral definition and a `lytic_component`
field on bone metastases) plus explicit aggressive-variant statements.

The rollup (tasks/avpc_adjacent/criteria.py) has no disease-state gate:
A1 any visceral / soft-tissue metastasis, A2 any lytic bone metastasis, A3
modified C5 (extent wording instead of a lesion count), A4 an explicit AVPC /
anaplastic / variant-CRPC statement, or any strict C1-C7 criterion without the
mCRPC requirement.

Outputs (under <output-dir>):
  avpc_adjacent_findings.parquet          grounded note-level facts
  avpc_adjacent_rejected_findings.parquet grounding/validation failures
  avpc_adjacent_processed_notes.parquet   per-note ledger, the unit of resume
  avpc_adjacent_run.parquet               extraction fingerprint
  avpc_adjacent_labels.parquet            one row per cohort patient; columns are
                                          avpc_adj_-prefixed except
                                          is_avpc_adjacent / avpc_adjacent_date
  avpc_adjacent_labels.meta.parquet       criteria and gate versions

Usage:
  python preprocessing/cli/collect_avpc_adjacent_notes.py --output-dir /path/to/out
  python tasks/avpc_adjacent/run_avpc_adjacent_extraction.py --output-dir /path/to/out --provider vertex_ai
  python tasks/avpc_adjacent/run_avpc_adjacent_extraction.py --output-dir /path/to/out --rebuild-labels-only
"""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import AVPC_ADJACENT_EVIDENCE_SCHEMA_VERSION, DEFAULT_DATA_PATH  # noqa: E402
from providers import get_provider  # noqa: E402
from tasks import note_extraction  # noqa: E402
from tasks.avpc import run_avpc_extraction as avpc_runner  # noqa: E402
from tasks.avpc.criteria import CRITERIA, CRITERIA_VERSION  # noqa: E402
from tasks.avpc_adjacent.criteria import (  # noqa: E402
    ADJ_CRITERIA,
    ADJ_CRITERIA_VERSION,
    ADJ_KEYS,
    COMPONENT_COLUMNS,
    evaluate_patient,
)
from tasks.avpc_adjacent.prompts import (  # noqa: E402
    AVPC_ADJACENT_CORRECTION_HINT,
    AVPC_ADJACENT_PROMPT_VERSION,
    AVPC_ADJACENT_SYSTEM_PROMPT,
    ENUMS,
    ITEM_FIELDS,
)
from tasks.mcrpc_gate import GATE_VERSION  # noqa: E402
from tasks.met_veto import VETO_VERSION  # noqa: E402
from tasks.nepc.ne_gate import NE_GATE_VERSION  # noqa: E402

PREFIX = "avpc_adj_"
# Kept unprefixed: the compiled primary_subtype label reads these directly.
UNPREFIXED = {"is_avpc_adjacent", "avpc_adjacent_date", "avpc_adjacent_date_precision"}

FINDING_SCHEMA = {
    **avpc_runner.FINDING_SCHEMA,
    "lytic_component": pl.Boolean,
    "term": pl.Utf8,
}
BOOL_FIELDS = avpc_runner.BOOL_FIELDS | {"lytic_component"}
FLOAT_FIELDS = avpc_runner.FLOAT_FIELDS
INT_FIELDS = avpc_runner.INT_FIELDS


def _column(name):
    return name if name in UNPREFIXED else PREFIX + name


_FIELDS = {
    "is_metastatic": pl.Boolean,
    "metastatic_date": pl.Utf8,
    "is_castration_resistant": pl.Boolean,
    "crpc_date": pl.Utf8,
    "is_mcrpc": pl.Boolean,
    "is_avpc_strict": pl.Boolean,
}
for _name in ADJ_CRITERIA:
    _key = ADJ_KEYS[_name]
    _FIELDS.update({
        f"{_key}_met": pl.Boolean,
        f"{_key}_quote": pl.Utf8,
        f"{_key}_date": pl.Utf8,
        f"{_key}_date_precision": pl.Utf8,
        f"{_key}_source_note_date": pl.Utf8,
    })
_FIELDS.update({column: pl.Utf8 for column in COMPONENT_COLUMNS})
for _name in CRITERIA:
    _key = _name.lower()
    _FIELDS.update({
        f"{_key}_met": pl.Boolean,
        f"{_key}_quote": pl.Utf8,
        f"{_key}_date": pl.Utf8,
        f"{_key}_date_precision": pl.Utf8,
        f"{_key}_source_note_date": pl.Utf8,
    })
_FIELDS.update({
    "criteria": pl.List(pl.Utf8),
    "n_criteria": pl.Int64,
    "is_avpc_adjacent": pl.Boolean,
    "avpc_adjacent_date": pl.Utf8,
    "avpc_adjacent_date_precision": pl.Utf8,
})

LABEL_SCHEMA = {"DFCI_MRN": pl.Int64, _column("review_status"): pl.Utf8}
LABEL_SCHEMA.update({_column(name): dtype for name, dtype in _FIELDS.items()})
LABEL_SCHEMA[_column("gate_rejections")] = pl.List(pl.Utf8)
BOOLEAN_FIELDS = [name for name, dtype in _FIELDS.items() if dtype == pl.Boolean]


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
                fields[name] = avpc_runner._as_float(raw)
            elif name in INT_FIELDS:
                number = avpc_runner._as_float(raw)
                if number is not None and not number.is_integer():
                    return None, f"invalid_{name}"
                fields[name] = None if number is None else int(number)
            elif name in BOOL_FIELDS:
                fields[name] = avpc_runner._as_bool(raw)
            elif name in ENUMS:
                value = raw.strip().lower() if isinstance(raw, str) else None
                # An unrecognized context is not an error, just not usable.
                fields[name] = value if value in ENUMS[name] else None
            else:
                fields[name] = avpc_runner._as_text(raw)
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
        if note_extraction.review_incomplete(status):
            # Unread notes are not evidence of absence: leave negatives unset.
            for name in BOOLEAN_FIELDS:
                if fields.get(name) is False:
                    fields[name] = None
        rows.append({
            "DFCI_MRN": mrn,
            _column("review_status"): status,
            **{_column(name): value for name, value in fields.items()},
            _column("gate_rejections"): rejections,
        })
    return rows


SPEC = note_extraction.TaskSpec(
    prefix="avpc_adjacent",
    label="AVPC-adjacent",
    system_prompt=AVPC_ADJACENT_SYSTEM_PROMPT,
    prompt_version=AVPC_ADJACENT_PROMPT_VERSION,
    evidence_schema_version=AVPC_ADJACENT_EVIDENCE_SCHEMA_VERSION,
    finding_schema=FINDING_SCHEMA,
    validate_finding=validate_finding,
    build_labels=build_labels,
    label_schema=LABEL_SCHEMA,
    label_versions={
        "adjacent_criteria_version": ADJ_CRITERIA_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "mcrpc_gate_version": GATE_VERSION,
        "met_veto_version": VETO_VERSION,
        "ne_gate_version": NE_GATE_VERSION,
        "review_policy_version": note_extraction.REVIEW_POLICY_VERSION,
    },
    correction_hint=AVPC_ADJACENT_CORRECTION_HINT,
    default_output_dir=Path(DEFAULT_DATA_PATH) / "LLM_avpc_adjacent",
    collector_hint="Run preprocessing/cli/collect_avpc_adjacent_notes.py first.",
)


def run(args):
    # Look the provider up at call time so tests can monkeypatch this module.
    return note_extraction.run(SPEC, args, lambda name: get_provider(name))


def main():
    run(note_extraction.build_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
