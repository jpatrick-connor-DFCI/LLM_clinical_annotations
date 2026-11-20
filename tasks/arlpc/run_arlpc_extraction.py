"""Stage 2 — Per-note AR-low / AR-indifferent (ARLPC) extraction and patient labels.

Reads arlpc_evidence.parquet from preprocessing/cli/collect_arlpc_notes.py and
makes ONE LLM call per evidence note. The model reports AR IHC results
(including AR-positive, for audit), AR-target IHC loss, AR-specific and
androgen-independent statements, and somatic genomic AR loss, each with a
verbatim quote and the date written in the text. Quotes are grounded against
their own note before storage; tasks/arlpc/ar_gate.py screens them at label
time.

Label tiers (independent of NE status and disease state; "ever AR-low" wins):
  strict  AR-low IHC (negative or any reduced wording), AR-specific statements,
          somatic genomic AR loss
  broad   strict + androgen-independent / hormone-refractory wording
AR-target IHC loss (PSA/PSAP/NKX3.1/PSMA/HOXB13) is corroborating only and
never sets a tier or a date.

Outputs (under <output-dir>):
  arlpc_findings.parquet          grounded note-level findings
  arlpc_rejected_findings.parquet grounding/validation failures with reasons
  arlpc_processed_notes.parquet   per-note ledger, the unit of resume
  arlpc_run.parquet               extraction fingerprint
  arlpc_labels.parquet            one row per cohort patient, arlpc_-prefixed
  arlpc_labels.meta.parquet       gate versions the labels used

Usage:
  python preprocessing/cli/collect_arlpc_notes.py --output-dir /path/to/out
  python tasks/arlpc/run_arlpc_extraction.py --output-dir /path/to/out --provider vertex_ai
  python tasks/arlpc/run_arlpc_extraction.py --output-dir /path/to/out --rebuild-labels-only
"""

import math
import re
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import ARLPC_EVIDENCE_SCHEMA_VERSION, DEFAULT_DATA_PATH  # noqa: E402
from providers import get_provider  # noqa: E402
from tasks import note_extraction  # noqa: E402
from tasks.arlpc.ar_gate import AR_GATE_VERSION, screen_ar_finding  # noqa: E402
from tasks.arlpc.prompts import (  # noqa: E402
    ARLPC_CORRECTION_HINT,
    ARLPC_PROMPT_VERSION,
    ARLPC_SYSTEM_PROMPT,
    ASSAYS,
    IHC_RESULTS,
    ITEM_FIELDS,
    ORIGINS,
    TARGET_MARKERS,
)
from tasks.met_veto import VETO_VERSION  # noqa: E402
from tasks.nepc.ne_gate import NE_GATE_VERSION  # noqa: E402
from tasks.onset import earliest  # noqa: E402

STRICT_TYPES = {"ar_ihc", "ar_statement", "ar_genomic_loss"}
BROAD_TYPES = STRICT_TYPES | {"androgen_independent_statement"}

FINDING_SCHEMA = {
    "evidence_type": pl.Utf8,
    "specimen_site": pl.Utf8,
    "ar_result": pl.Utf8,
    "grade_text": pl.Utf8,
    "ar_percent": pl.Float64,
    "ar_h_score": pl.Float64,
    "marker": pl.Utf8,
    "marker_result": pl.Utf8,
    "assay": pl.Utf8,
    "favored_origin": pl.Utf8,
}
ENUM_FIELDS = {
    "ar_result": IHC_RESULTS,
    "marker_result": IHC_RESULTS,
    "marker": TARGET_MARKERS,
    "assay": ASSAYS,
    "favored_origin": ORIGINS,
}
NUMBER_RANGES = {"ar_percent": (0.0, 100.0), "ar_h_score": (0.0, 300.0)}
MARKER_ALIASES = {
    "nkx3.1": "nkx3_1", "nkx3-1": "nkx3_1", "nkx31": "nkx3_1", "nkx 3.1": "nkx3_1",
    "prostatic acid phosphatase": "psap", "pap": "psap",
    "prostate-specific antigen": "psa", "prostate specific antigen": "psa",
}

LABEL_SCHEMA = {"DFCI_MRN": pl.Int64, "arlpc_review_status": pl.Utf8}
for _tier in ("strict", "broad"):
    LABEL_SCHEMA.update({
        f"arlpc_{_tier}": pl.Boolean,
        f"arlpc_{_tier}_date": pl.Utf8,
        f"arlpc_{_tier}_date_precision": pl.Utf8,
        f"arlpc_{_tier}_date_source": pl.Utf8,
        f"arlpc_{_tier}_evidence_type": pl.Utf8,
        f"arlpc_{_tier}_quote": pl.Utf8,
        f"arlpc_{_tier}_source_note_date": pl.Utf8,
        f"arlpc_{_tier}_note_type": pl.Utf8,
    })
LABEL_SCHEMA.update({
    "arlpc_tissue_date": pl.Utf8,
    "arlpc_tissue_date_precision": pl.Utf8,
    "arlpc_tissue_date_source": pl.Utf8,
    "arlpc_tissue_quote": pl.Utf8,
    "arlpc_has_ar_target_loss": pl.Boolean,
    "arlpc_ar_target_markers": pl.List(pl.Utf8),
    "arlpc_ar_target_date": pl.Utf8,
    "arlpc_ar_target_quote": pl.Utf8,
    "arlpc_n_ar_positive_ihc": pl.Int64,
    "arlpc_evidence_types": pl.List(pl.Utf8),
    "arlpc_n_gated_findings": pl.Int64,
    "arlpc_gate_rejections": pl.List(pl.Utf8),
})


def _as_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "").rstrip("%").strip()
        if not value or value.lower() in {"null", "none"}:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError from None
    if not math.isfinite(number):
        raise ValueError
    return number


def _as_text(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v).strip() for v in value if str(v).strip())
    value = str(value).strip()
    return value or None


def validate_finding(finding, note):
    evidence_type = finding.get("evidence_type")
    if isinstance(evidence_type, str):
        evidence_type = evidence_type.strip().lower()
    if evidence_type not in ITEM_FIELDS:
        return None, f"invalid_evidence_type:{evidence_type}"
    fields = {column: None for column in FINDING_SCHEMA}
    fields["evidence_type"] = evidence_type
    for name in ITEM_FIELDS[evidence_type]:
        raw = finding.get(name)
        if name in NUMBER_RANGES:
            try:
                number = _as_number(raw)
            except ValueError:
                return None, f"invalid_{name}"
            lo, hi = NUMBER_RANGES[name]
            if number is not None and not lo <= number <= hi:
                return None, f"invalid_{name}"
            fields[name] = number
        elif name in ENUM_FIELDS:
            value = re.sub(r"\s+", " ", raw.strip().lower()) if isinstance(raw, str) else None
            if name == "marker":
                value = MARKER_ALIASES.get(value, value)
            # An unrecognized value is not an error, just not usable.
            fields[name] = value if value in ENUM_FIELDS[name] else None
        else:
            fields[name] = _as_text(raw)
    return fields, None


def gate_findings(findings):
    """Split findings into ({mrn: [gated rows]}, {mrn: [rejection reasons]})."""
    gated, rejections = {}, {}
    for row in findings.iter_rows(named=True):
        mrn = int(row["DFCI_MRN"])
        trial = row.get("_trial_context_reason")
        if trial:
            ok, reason, info = False, trial, {}
        else:
            ok, reason, info = screen_ar_finding(row)
        if ok:
            gated.setdefault(mrn, []).append({**row, **info})
        else:
            rejections.setdefault(mrn, []).append(f"{row['evidence_type']}:{reason}")
    return gated, rejections


def _tag(row):
    if row["evidence_type"] == "ar_ihc":
        return "ar_ihc" if row.get("ar_low") else "ar_ihc_positive"
    return row["evidence_type"]


def _tier_fields(tier, first, labeled):
    return {
        f"arlpc_{tier}": (first is not None) if labeled else (True if first else None),
        f"arlpc_{tier}_date": first and first["event_date"],
        f"arlpc_{tier}_date_precision": first and first["date_precision"],
        f"arlpc_{tier}_date_source": first and first["date_source"],
        f"arlpc_{tier}_evidence_type": first and first["evidence_type"],
        f"arlpc_{tier}_quote": first and first["quote"],
        f"arlpc_{tier}_source_note_date": first and first["source_note_date"],
        f"arlpc_{tier}_note_type": first and first["note_type"],
    }


def build_labels(findings, patient_status):
    gated, rejections = gate_findings(findings)
    rows = []
    for mrn, status in patient_status.items():
        mine = gated.get(mrn, [])
        labeled = not note_extraction.review_incomplete(status)
        ar_low = [r for r in mine if r["evidence_type"] == "ar_ihc" and r.get("ar_low")]
        setting = ar_low + [r for r in mine if r["evidence_type"] in BROAD_TYPES - {"ar_ihc"}]
        strict = earliest([r for r in setting if r["evidence_type"] in STRICT_TYPES])
        broad = earliest(setting)
        tissue = earliest(ar_low)
        targets = [r for r in mine if r["evidence_type"] == "ar_target_ihc"]
        first_target = earliest(targets)
        has_target = bool(targets)
        rows.append({
            "DFCI_MRN": mrn,
            "arlpc_review_status": status,
            **_tier_fields("strict", strict, labeled),
            **_tier_fields("broad", broad, labeled),
            "arlpc_tissue_date": tissue and tissue["event_date"],
            "arlpc_tissue_date_precision": tissue and tissue["date_precision"],
            "arlpc_tissue_date_source": tissue and tissue["date_source"],
            "arlpc_tissue_quote": tissue and tissue["quote"],
            "arlpc_has_ar_target_loss": has_target if (labeled or has_target) else None,
            "arlpc_ar_target_markers": sorted({m for r in targets for m in r["markers"]}),
            "arlpc_ar_target_date": first_target and first_target["event_date"],
            "arlpc_ar_target_quote": first_target and first_target["quote"],
            "arlpc_n_ar_positive_ihc": sum(
                1 for r in mine if r["evidence_type"] == "ar_ihc" and not r.get("ar_low")
            ),
            "arlpc_evidence_types": sorted({_tag(r) for r in mine}),
            "arlpc_n_gated_findings": len(mine),
            "arlpc_gate_rejections": sorted(set(rejections.get(mrn, []))),
        })
    return rows


SPEC = note_extraction.TaskSpec(
    prefix="arlpc",
    label="ARLPC",
    system_prompt=ARLPC_SYSTEM_PROMPT,
    prompt_version=ARLPC_PROMPT_VERSION,
    evidence_schema_version=ARLPC_EVIDENCE_SCHEMA_VERSION,
    finding_schema=FINDING_SCHEMA,
    validate_finding=validate_finding,
    build_labels=build_labels,
    label_schema=LABEL_SCHEMA,
    label_versions={
        "ar_gate_version": AR_GATE_VERSION,
        "review_policy_version": note_extraction.REVIEW_POLICY_VERSION,
        "ne_gate_version": NE_GATE_VERSION,
        "met_veto_version": VETO_VERSION,
    },
    correction_hint=ARLPC_CORRECTION_HINT,
    default_output_dir=Path(DEFAULT_DATA_PATH) / "LLM_arlpc",
    collector_hint="Run preprocessing/cli/collect_arlpc_notes.py first.",
)


def run(args):
    # Look the provider up at call time so tests can monkeypatch this module.
    return note_extraction.run(SPEC, args, lambda name: get_provider(name))


def main():
    run(note_extraction.build_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
