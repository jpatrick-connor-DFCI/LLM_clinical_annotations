"""Build mCRPC diagnosis-date labels from the disease-course extraction (no LLM calls).

Combines the course findings (tasks/disease_course/run_course_extraction.py),
LABS PSA and testosterone, MEDICATIONS, and an upstream metastatic label
(--met-source avpc, the default, or met_dx), applies tasks/mcrpc/criteria.py,
and writes:

  mcrpc_labels.parquet       one row per cohort patient: mcrpc_date and basis
                             (both_hold | statement | drug_proxy), the primary
                             basis, proxy and CRPC audit columns
  mcrpc_labels.meta.parquet  versions, course run, structured sources, rule
                             defaults, and the upstream label's path, sha256
                             and metadata (a label-only upstream rebuild can
                             move the metastatic date)

Usage:
  python tasks/mcrpc/build_mcrpc_labels.py --output-dir /path/to/course_out
  python tasks/mcrpc/build_mcrpc_labels.py --output-dir /path/to/course_out \
      --met-source met_dx --met-labels /path/to/met_dx_labels.parquet
"""

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_DATA_PATH  # noqa: E402
from tasks.disease_course import builder  # noqa: E402
from tasks.mcrpc.criteria import RULE_DEFAULTS, evaluate_mcrpc  # noqa: E402

# source -> (default labels path, flag column, date column, precision column)
MET_SOURCES = {
    "avpc": (Path(DEFAULT_DATA_PATH) / "LLM_avpc" / "avpc_labels.parquet",
             "is_metastatic", "metastatic_date", "metastatic_date_precision"),
    "met_dx": (Path(DEFAULT_DATA_PATH) / "LLM_met_diagnosis" / "met_dx_labels.parquet",
               "has_metastatic_disease", "first_metastasis_date", "date_precision"),
}

LIST_COLUMNS = {"crpc_testosterone_dates", "pcwg3_unevaluable", "gate_rejections"}
BOOL_COLUMNS = {"is_mcrpc", "proxy_precedes_primary"}
COLUMNS = [
    "is_mcrpc", "mcrpc_date", "mcrpc_date_precision", "mcrpc_basis", "proxy_precedes_primary",
    "primary_date", "primary_date_precision", "primary_basis",
    "met_date", "met_date_precision", "met_label_source",
    "crpc_date", "crpc_date_precision", "crpc_basis", "crpc_source", "crpc_quote",
    "crpc_castration_tier", "crpc_testosterone_dates", "crpc_exposure_basis",
    "crpc_exposure_certainty", "crpc_confirmation_date", "pcwg3_unevaluable",
    "mcrpc_statement_date", "mcrpc_statement_precision", "mcrpc_statement_quote",
    "proxy_date", "proxy_date_precision", "proxy_agent", "proxy_evidence", "proxy_source",
    "lu_psma_date",
]
LABEL_SCHEMA = {"DFCI_MRN": pl.Int64, "review_status": pl.Utf8}
LABEL_SCHEMA.update({
    c: pl.List(pl.Utf8) if c in LIST_COLUMNS else pl.Boolean if c in BOOL_COLUMNS else pl.Utf8
    for c in COLUMNS
})
LABEL_SCHEMA["gate_rejections"] = pl.List(pl.Utf8)


def load_met_labels(source, path=None):
    """{mrn: {"date", "precision", "source"}} for patients the upstream label calls metastatic."""
    default, flag, date_column, precision_column = MET_SOURCES[source]
    path = Path(path or default)
    if not path.exists():
        raise FileNotFoundError(
            f"{source} labels not found at {path}; build them first or pass --met-labels."
        )
    frame = pl.read_parquet(path)
    missing = {"DFCI_MRN", flag, date_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)} for --met-source {source}")
    has_precision = precision_column in frame.columns
    labels = {}
    for row in frame.iter_rows(named=True):
        if row[flag] is not True or not row[date_column]:
            continue
        labels[int(row["DFCI_MRN"])] = {
            "date": row[date_column],
            "precision": row[precision_column] if has_precision else "unknown",
            "source": source,
        }
    return labels, path


def build(inputs, structured, met_labels):
    rows = []
    for mrn in sorted(inputs["status"]):
        status = inputs["status"][mrn]
        fields = evaluate_mcrpc(
            inputs["usable"].get(mrn, []),
            structured["psa"].get(mrn, []),
            structured["medications"].get(mrn, []),
            structured["testosterone"].get(mrn, []),
            met_labels.get(mrn),
        )
        fields = builder.blank_unreviewed(fields, status)
        rows.append({
            "DFCI_MRN": mrn,
            "review_status": status,
            **fields,
            "gate_rejections": inputs["rejected"].get(mrn, []),
        })
    return rows


def main(argv=None):
    parser = builder.build_parser("Build mCRPC labels from disease-course findings.")
    parser.add_argument("--met-source", choices=sorted(MET_SOURCES), default="avpc")
    parser.add_argument("--met-labels", type=Path, default=None,
                        help="Upstream metastatic label parquet (default per --met-source).")
    args = parser.parse_args(argv)
    inputs = builder.load_course(args)
    met_labels, met_path = load_met_labels(args.met_source, args.met_labels)
    structured, fingerprint = builder.load_structured(
        ["psa", "testosterone", "medications"], inputs["cohort"], args.profile_data_path
    )
    rows = build(inputs, structured, met_labels)
    metadata = builder.base_metadata(
        inputs, fingerprint,
        mcrpc_rule_defaults=json.dumps(RULE_DEFAULTS, sort_keys=True),
        met_source=args.met_source,
        met_labels=json.dumps(builder.upstream_identity(met_path), sort_keys=True),
    )
    path = args.output_dir / "mcrpc_labels.parquet"
    labels = builder.write_output(path, rows, LABEL_SCHEMA, metadata)
    print(f"Wrote mCRPC labels ({labels.height} patients, met source {args.met_source}): {path}")
    print(labels.group_by("mcrpc_basis", "proxy_precedes_primary").len().sort("mcrpc_basis", nulls_last=True))
    return labels


if __name__ == "__main__":
    main()
