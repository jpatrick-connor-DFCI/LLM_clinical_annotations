"""Shared plumbing for the BCR, progression and mCRPC builders (no LLM calls).

Each builder reads the finished disease-course extraction through
`tasks.note_extraction.load_label_inputs`, screens every finding through
`statement_gate.usable_course_findings` (so trial-context and gate rejections
reach every endpoint, PSA and therapy facts included), loads the structured
PROFILE_DATA sources it needs, and writes `<name>.parquet` beside a
`<name>.meta.parquet` recording every version and rule default it used.
"""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from preprocessing.config import PROFILE_DATA_PATH
from preprocessing.parquet_io import read_metadata, write_metadata, write_parquet_atomic
from preprocessing.structured import (
    VITAL_STATUS_POLICY,
    load_medications,
    load_psa,
    load_testosterone,
    load_vital_status,
    source_path,
    structured_source_fingerprint,
)
from tasks.disease_course import run_course_extraction as course_runner
from tasks.disease_course.psa import RULE_DEFAULTS
from tasks.disease_course.statement_gate import usable_course_findings
from tasks.disease_course.therapy import THERAPY_VERSION
from tasks.note_extraction import load_label_inputs
from tasks.trial_guard import TRIAL_GUARD_VERSION

UNLABELED_STATUSES = course_runner.UNLABELED_STATUSES
STRUCTURED_SOURCES = {"psa": "labs", "testosterone": "labs", "medications": "medications",
                      "vital_status": "registry"}
_LOADERS = {"psa": load_psa, "testosterone": load_testosterone,
            "medications": load_medications, "vital_status": load_vital_status}


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=course_runner.DEFAULT_OUTPUT_DIR,
                        help="Disease-course extraction directory; labels are written here too.")
    parser.add_argument("--evidence-path", type=Path, default=None)
    parser.add_argument("--profile-data-path", type=Path, default=Path(PROFILE_DATA_PATH),
                        help="Directory holding LABS / MEDICATIONS / registry parquets.")
    return parser


def load_course(args):
    """Course inputs plus gated findings: {..., "usable", "rejected"}."""
    inputs = load_label_inputs(course_runner.SPEC, args.output_dir, args.evidence_path)
    inputs["usable"], inputs["rejected"] = usable_course_findings(inputs["findings"])
    return inputs


def load_structured(names, mrns, root):
    """({name: {mrn: [row dicts]}}, fingerprint) for the requested structured sources."""
    out, paths = {}, {}
    for name in names:
        path = source_path(STRUCTURED_SOURCES[name], root)
        paths[STRUCTURED_SOURCES[name]] = path
        frame = _LOADERS[name](mrns, path)
        grouped = {}
        for row in frame.iter_rows(named=True):
            grouped.setdefault(int(row["DFCI_MRN"]), []).append(row)
        out[name] = grouped
    return out, structured_source_fingerprint(paths)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_metadata(inputs, structured_fingerprint, **versions):
    return {
        "course_run_config": inputs["run_config"],
        "course_label_versions": json.dumps(
            {k: str(v) for k, v in course_runner.SPEC.label_versions.items()}, sort_keys=True
        ),
        "trial_guard_version": TRIAL_GUARD_VERSION,
        "therapy_version": THERAPY_VERSION,
        "vital_status_policy": VITAL_STATUS_POLICY,
        "structured_sources": structured_fingerprint,
        "psa_rule_defaults": json.dumps(RULE_DEFAULTS, sort_keys=True),
        **{k: (v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str))
           for k, v in versions.items()},
    }


def frame(rows, schema):
    return pl.DataFrame({c: [row.get(c) for row in rows] for c in schema}, schema=schema, strict=False)


def write_output(path, rows, schema, metadata=None):
    path = Path(path)
    data = frame(rows, schema)
    write_parquet_atomic(data, path)
    if metadata is not None:
        write_metadata(path.with_name(f"{path.stem}.meta.parquet"), metadata)
    return data


def upstream_identity(path):
    """sha256 and stored metadata of an upstream label artifact."""
    path = Path(path)
    meta = read_metadata(path.with_name(f"{path.stem}.meta.parquet"))
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "meta": json.dumps(meta, sort_keys=True, default=str) if meta else None,
    }


def blank_unreviewed(fields, status, keep=()):
    """Null every note-derived field when the review failed or never ran."""
    if status not in UNLABELED_STATUSES:
        return fields
    return {k: (v if k in keep else None) for k, v in fields.items()}
