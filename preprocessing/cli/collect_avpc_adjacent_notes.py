"""Stage 1 — Collect notes that may document an AVPC-adjacent criterion.

AVPC-adjacent (tasks/avpc_adjacent) is a relaxed, ungated companion to the
strict Aparicio pipeline: any visceral (extended site list) metastasis, any
lytic bone metastasis, a modified C5 that accepts extent wording, an explicit
"aggressive variant" / anaplastic / variant-CRPC statement, or any strict
C1-C7 criterion without the mCRPC state requirement. It re-extracts the strict
facts itself, so its triggers are the full AVPC set plus the extended sites
and extent wording (preprocessing/triggers.py::AVPC_ADJACENT_TRIGGER_REGEX).

Writes one evidence row per deduplicated note; the task runner
(tasks/avpc_adjacent/run_avpc_adjacent_extraction.py) makes one LLM call per
note. Check the per-note-type size summary this prints before the LLM run.

Outputs (under <output-dir>):
  avpc_adjacent_evidence.parquet        one row per note: DFCI_MRN, note_uid,
                                        note_date, note_type, snippet
  avpc_adjacent_evidence.meta.parquet   scan_config hash, evidence sha256, cohort

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
    AVPC_ADJACENT_EVIDENCE_SCHEMA_VERSION,
    DEFAULT_PROSTATE_MRN_CSV,
    DEFAULT_DATA_PATH,
    SNIPPET_PROFILES,
)
from preprocessing.evidence import (  # noqa: E402
    add_collector_args,
    collect_note_evidence,
    downstream_paths,
    note_task_paths,
)
from preprocessing.triggers import AVPC_ADJACENT_TRIGGER_REGEX  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_avpc_adjacent"
TASK_PREFIX = "avpc_adjacent"
TRIGGER_REGEX = AVPC_ADJACENT_TRIGGER_REGEX
# Wide enough to carry a measurement, a lesion count, or a date next to a hit.
DEFAULT_CONTEXT_CHARS = 2000
SNIPPET_MAX_CHARS = SNIPPET_PROFILES["longitudinal"].max_chars


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect notes that may document an AVPC-adjacent criterion."
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
        schema_version=AVPC_ADJACENT_EVIDENCE_SCHEMA_VERSION,
        snippet_max_chars=SNIPPET_MAX_CHARS,
        downstream_paths=downstream_paths(args.output_dir, TASK_PREFIX),
        label="AVPC-adjacent",
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
