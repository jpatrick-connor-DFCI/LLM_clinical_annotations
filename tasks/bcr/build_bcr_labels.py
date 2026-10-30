"""Build biochemical-recurrence labels from the disease-course extraction (no LLM calls).

Reads the course findings (tasks/disease_course/run_course_extraction.py) and
LABS PSA, applies tasks/bcr/criteria.py, and writes one row per cohort patient:

  bcr_labels.parquet       bcr, bcr_date, bcr_basis (psa_aua | psa_phoenix |
                           statement | imaging), index date, persistence,
                           clearance, PSA-rule status and audit columns
  bcr_labels.meta.parquet  versions, course run, structured sources, rule defaults

Usage:
  python tasks/bcr/build_bcr_labels.py --output-dir /path/to/course_out
"""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.bcr.criteria import BCR_CRITERIA_VERSION, evaluate_bcr  # noqa: E402
from tasks.disease_course import builder  # noqa: E402

LIST_COLUMNS = {"bcr_bases", "psa_bcr_source_ids", "imaging_exclusions", "psa_audit", "gate_rejections"}
FLOAT_COLUMNS = {"psa_bcr_value", "phoenix_nadir"}
BOOL_COLUMNS = {"bcr", "persistent_psa"}
INT_COLUMNS = {"n_psa_measurements", "n_psa_conflicts"}
COLUMNS = [
    "bcr", "bcr_date", "bcr_date_precision", "bcr_basis", "bcr_bases", "bcr_source",
    "local_therapy", "index_date", "index_date_precision",
    "persistent_psa", "persistent_psa_basis", "persistent_psa_reason", "psa_clearance_date",
    "psa_bcr_rule", "psa_bcr_status", "psa_bcr_reason", "psa_bcr_date", "psa_bcr_value",
    "psa_bcr_confirmation_date", "psa_bcr_source_ids", "phoenix_nadir",
    "statement_date", "statement_date_precision", "statement_quote",
    "imaging_date", "imaging_date_precision", "imaging_certainty", "imaging_modality",
    "imaging_site", "imaging_quote", "imaging_exclusions",
    "n_psa_measurements", "n_psa_conflicts", "psa_audit",
]


def _dtype(column):
    if column in LIST_COLUMNS:
        return pl.List(pl.Utf8)
    if column in FLOAT_COLUMNS:
        return pl.Float64
    if column in BOOL_COLUMNS:
        return pl.Boolean
    if column in INT_COLUMNS:
        return pl.Int64
    return pl.Utf8


LABEL_SCHEMA = {"DFCI_MRN": pl.Int64, "review_status": pl.Utf8}
LABEL_SCHEMA.update({c: _dtype(c) for c in COLUMNS})
LABEL_SCHEMA["gate_rejections"] = pl.List(pl.Utf8)


def build(inputs, structured):
    rows = []
    for mrn in sorted(inputs["status"]):
        status = inputs["status"][mrn]
        fields = evaluate_bcr(inputs["usable"].get(mrn, []), structured["psa"].get(mrn, []))
        fields = builder.blank_unreviewed(fields, status)
        rows.append({
            "DFCI_MRN": mrn,
            "review_status": status,
            **fields,
            "gate_rejections": inputs["rejected"].get(mrn, []),
        })
    return rows


def main(argv=None):
    args = builder.build_parser("Build BCR labels from disease-course findings.").parse_args(argv)
    inputs = builder.load_course(args)
    structured, fingerprint = builder.load_structured(
        ["psa"], inputs["cohort"], args.profile_data_path
    )
    rows = build(inputs, structured)
    path = args.output_dir / "bcr_labels.parquet"
    labels = builder.write_output(
        path, rows, LABEL_SCHEMA,
        builder.base_metadata(inputs, fingerprint, bcr_criteria_version=BCR_CRITERIA_VERSION),
    )
    counts = labels.group_by("bcr_basis").len().sort("bcr_basis", nulls_last=True)
    print(f"Wrote BCR labels ({labels.height} patients): {path}")
    print(counts)
    return labels


if __name__ == "__main__":
    main()
