"""Stage 1 — Collect notes that may document neuroendocrine prostate cancer.

Scans every cohort patient's notes for neuroendocrine / small-cell language and
NE IHC markers and writes one evidence row per deduplicated note. The NEPC task
runner (tasks/nepc/run_nepc_extraction.py) then makes one LLM call per note.

Trigger-bearing Pathology reports are kept whole (capped at the per-note
limit): the specimen site in the header and the IHC panel in the addendum are
what show a finding is prostate-derived, and they are often far from the
trigger word. Imaging and clinician notes keep a window around each hit.

Outputs (under <output-dir>):
  nepc_evidence.parquet        one row per note: DFCI_MRN, note_uid, note_date,
                               note_type, snippet
  nepc_evidence.meta.parquet   scan_config hash, evidence sha256, cohort MRNs

Re-running with unchanged settings reuses the evidence; changed settings raise
until --overwrite is passed, which also discards the runner's outputs.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import (  # noqa: E402
    DEFAULT_PROSTATE_MRN_CSV,
    DEFAULT_DATA_PATH,
    NEPC_DX_EVIDENCE_SCHEMA_VERSION,
    SNIPPET_PROFILES,
)
from preprocessing.evidence import (  # noqa: E402
    add_collector_args,
    collect_note_evidence,
    downstream_paths,
    note_task_paths,
)
from preprocessing.triggers import NEPC_DX_TRIGGER_REGEX  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_nepc"
TASK_PREFIX = "nepc"
TRIGGER_REGEX = NEPC_DX_TRIGGER_REGEX
DEFAULT_CONTEXT_CHARS = 1500
SNIPPET_MAX_CHARS = SNIPPET_PROFILES["longitudinal"].max_chars
# None = keep the whole cleaned report for this note type.
CONTEXT_OVERRIDES = {"Pathology": None}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect notes that may document neuroendocrine prostate cancer."
    )
    add_collector_args(
        parser,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        default_mrn_file=DEFAULT_PROSTATE_MRN_CSV,
        default_context_chars=DEFAULT_CONTEXT_CHARS,
    )
    args = parser.parse_args()
    if args.context_chars < 0:
        parser.error("--context-chars must be >= 0")
    if args.scan_workers is not None and args.scan_workers < 1:
        parser.error("--scan-workers must be >= 1")
    return args


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return collect_note_evidence(
        args,
        trigger_regex=TRIGGER_REGEX,
        evidence_path=note_task_paths(args.output_dir, TASK_PREFIX)["evidence"],
        schema_version=NEPC_DX_EVIDENCE_SCHEMA_VERSION,
        snippet_max_chars=SNIPPET_MAX_CHARS,
        downstream_paths=downstream_paths(args.output_dir, TASK_PREFIX),
        context_overrides=CONTEXT_OVERRIDES,
        label="NEPC",
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
