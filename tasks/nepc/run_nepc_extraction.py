"""Stage 2 — Per-note NEPC extraction and patient-level NEPC labels.

Reads nepc_evidence.parquet from preprocessing/cli/collect_nepc_dx_notes.py and
makes ONE LLM call per evidence note. The model reports atomic neuroendocrine
findings (stated diagnosis, histologic transformation, NE differentiation,
positive NE IHC), each with a verbatim quote and the date written in the text.
Quotes are grounded against their own note before storage.

Labels are a pure function of the stored findings: every finding must pass the
deterministic NE gate (tasks/nepc/ne_gate.py), and a patient is "nepc" when at
least one does. The cited quote and date come from the earliest gated finding
(tasks/onset.py: a certainly-earlier date wins; within overlapping dates the
more precise one wins). `has_stated_nepc_dx` is the strict tier for sensitivity
analyses: a stated diagnosis or transformation, not differentiation or IHC
alone.

Outputs (under <output-dir>):
  nepc_findings.parquet          grounded note-level findings
  nepc_rejected_findings.parquet grounding/validation failures with reasons
  nepc_processed_notes.parquet   per-note ledger, the unit of resume
  nepc_run.parquet               extraction fingerprint
  nepc_labels.parquet            one row per cohort patient
  nepc_labels.meta.parquet       gate version the labels used

Usage:
  python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/out
  python tasks/nepc/run_nepc_extraction.py --output-dir /path/to/out --provider vertex_ai
  # Re-apply a changed gate without LLM calls:
  python tasks/nepc/run_nepc_extraction.py --output-dir /path/to/out --rebuild-labels-only
"""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_DATA_PATH, NEPC_DX_EVIDENCE_SCHEMA_VERSION  # noqa: E402
from providers import get_provider  # noqa: E402
from tasks import note_extraction  # noqa: E402
from tasks.nepc.ne_gate import NE_GATE_VERSION, screen_ne_quote  # noqa: E402
from tasks.nepc.prompts import (  # noqa: E402
    EVIDENCE_TYPES,
    NEPC_CORRECTION_HINT,
    NEPC_PROMPT_VERSION,
    NEPC_SYSTEM_PROMPT,
)
from tasks.onset import earliest  # noqa: E402

STATED_DX_TYPES = {"nepc_diagnosis", "histologic_transformation"}
# Patients whose notes were never read have no label, not a negative one.
UNLABELED_STATUSES = {"failed", "not_run"}

FINDING_SCHEMA = {
    "evidence_type": pl.Utf8,
    "specimen_site": pl.Utf8,
}
LABEL_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "review_status": pl.Utf8,
    "nepc_status": pl.Utf8,
    "nepc_date": pl.Utf8,
    "nepc_date_source": pl.Utf8,
    "nepc_date_precision": pl.Utf8,
    "nepc_evidence_type": pl.Utf8,
    "nepc_quote": pl.Utf8,
    "nepc_source_note_date": pl.Utf8,
    "nepc_note_type": pl.Utf8,
    "has_stated_nepc_dx": pl.Boolean,
    "stated_nepc_dx_date": pl.Utf8,
    "stated_nepc_dx_date_precision": pl.Utf8,
    "stated_nepc_dx_quote": pl.Utf8,
    "evidence_types": pl.List(pl.Utf8),
    "n_gated_findings": pl.Int64,
    "gate_rejections": pl.List(pl.Utf8),
}


def validate_finding(finding, note):
    evidence_type = finding.get("evidence_type")
    if isinstance(evidence_type, str):
        evidence_type = evidence_type.strip().lower()
    if evidence_type not in EVIDENCE_TYPES:
        return None, f"invalid_evidence_type:{evidence_type}"
    site = finding.get("specimen_site")
    if site is not None and not isinstance(site, str):
        return None, "invalid_specimen_site"
    site = site.strip() if isinstance(site, str) and site.strip() else None
    return {"evidence_type": evidence_type, "specimen_site": site}, None


def gate_findings(findings):
    """Split findings into ({mrn: [gated rows]}, {mrn: [rejection reasons]})."""
    gated, rejections = {}, {}
    for row in findings.iter_rows(named=True):
        mrn = int(row["DFCI_MRN"])
        reason = row.get("_trial_context_reason")
        ok, reason = (False, reason) if reason else screen_ne_quote(row["quote"], row["evidence_type"])
        if ok:
            gated.setdefault(mrn, []).append(row)
        else:
            rejections.setdefault(mrn, []).append(f"{row['evidence_type']}:{reason}")
    return gated, rejections


def build_labels(findings, patient_status):
    gated, rejections = gate_findings(findings)
    rows = []
    for mrn, status in patient_status.items():
        mine = gated.get(mrn, [])
        first = earliest(mine)
        stated = earliest([r for r in mine if r["evidence_type"] in STATED_DX_TYPES])
        if first is not None:
            nepc_status = "nepc"
        elif status in UNLABELED_STATUSES:
            nepc_status = None
        else:
            nepc_status = "non_nepc"
        rows.append({
            "DFCI_MRN": mrn,
            "review_status": status,
            "nepc_status": nepc_status,
            "nepc_date": first and first["event_date"],
            "nepc_date_source": first and first["date_source"],
            "nepc_date_precision": first and first["date_precision"],
            "nepc_evidence_type": first and first["evidence_type"],
            "nepc_quote": first and first["quote"],
            "nepc_source_note_date": first and first["source_note_date"],
            "nepc_note_type": first and first["note_type"],
            "has_stated_nepc_dx": stated is not None if nepc_status is not None else None,
            "stated_nepc_dx_date": stated and stated["event_date"],
            "stated_nepc_dx_date_precision": stated and stated["date_precision"],
            "stated_nepc_dx_quote": stated and stated["quote"],
            "evidence_types": sorted({r["evidence_type"] for r in mine}),
            "n_gated_findings": len(mine),
            "gate_rejections": sorted(set(rejections.get(mrn, []))),
        })
    return rows


SPEC = note_extraction.TaskSpec(
    prefix="nepc",
    label="NEPC",
    system_prompt=NEPC_SYSTEM_PROMPT,
    prompt_version=NEPC_PROMPT_VERSION,
    evidence_schema_version=NEPC_DX_EVIDENCE_SCHEMA_VERSION,
    finding_schema=FINDING_SCHEMA,
    validate_finding=validate_finding,
    build_labels=build_labels,
    label_schema=LABEL_SCHEMA,
    label_versions={"ne_gate_version": NE_GATE_VERSION},
    correction_hint=NEPC_CORRECTION_HINT,
    default_output_dir=Path(DEFAULT_DATA_PATH) / "LLM_nepc",
    collector_hint="Run preprocessing/cli/collect_nepc_dx_notes.py first.",
)


def run(args):
    # Look the provider up at call time so tests can monkeypatch this module.
    return note_extraction.run(SPEC, args, lambda name: get_provider(name))


def main():
    run(note_extraction.build_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
