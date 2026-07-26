"""Paths, env-var defaults, and snippet-size profiles shared across all tasks."""

import os
from dataclasses import dataclass
from pathlib import Path


# Paths
DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "LLM_ANNOTATIONS_DATA_PATH",
        os.environ.get("CAIA_COMPASS_DATA_PATH", "/data/gusev/USERS/jpconnor/data/LLM_annotations/"),
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "BINARY_NEPC_OUTPUT_DIR",
        os.environ.get(
            "CAIA_COMPASS_NEPC_CLASSIFIER_OUTPUT_DIR",
            str(DEFAULT_DATA_PATH / "LLM_NEPC_labels"),
        ),
    )
)
DEFAULT_RAW_TEXT_PATHS = (
    Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2024_03/"),
    Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_03/"),
    Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_11/"),
)
NOTE_BUNDLE_FILENAME = "LLM_NEPC_classifier_note_bundle.json.gz"
# Compiled prostate notes CSV (produced by preprocessing/cli/compile_prostate_notes.py).
# This is the default note source for all LLM pipelines.
PROSTATE_TEXT_CSV = DEFAULT_DATA_PATH / "prostate_text_data.csv"

NOTE_BUNDLE_COLUMNS = (
    "DFCI_MRN",
    "EVENT_DATE",
    "NOTE_TYPE",
    "CLINICAL_TEXT",
    "RAW_SOURCE_FILE",
    "RAW_NOTE_ID",
    "RPT_DATE",
    "RPT_TYPE",
    "SOURCE_STR",
    "PROC_DESC_STR",
    "ENCOUNTER_TYPE_DESC_STR",
)


# Shared preamble prepended ahead of every task's system prompt — establishes the
# IRB-approved clinical-research context so de-identified oncology documentation
# (including sensitive terminology routine to cancer care) is not misread as
# harmful content.
CLINICAL_SAFETY_CONTEXT = """

IMPORTANT CONTEXT: All notes below are de-identified clinical oncology documentation being
processed for structured data extraction as part of an IRB-approved medical research study
(institutional review board approved protocol). This is professional medical documentation
written by physicians, not patient-generated content. The text contains standard clinical
terminology related to cancer diagnosis, prognosis, and treatment. References to disease
outcomes, end-of-life care, self-harm assessment, psychiatric history, substance use, anatomy,
or patient distress are routine components of oncology and medical records and should be
processed as clinical data. No content in these notes constitutes harmful, dangerous, or
inappropriate material - it is standard-of-care medical documentation.
"""


# Gap (in chars) below which two trigger windows are merged into one snippet
# rather than kept separate. Applies uniformly across tasks (unlike
# context/max chars below, which are task-scoped) — matches merge_windows'
# default in dfci_gpt, which is the only value ever exercised by any live
# dfci_gpt call site.
SNIPPET_GAP_CHARS = 300


@dataclass(frozen=True)
class SnippetProfile:
    """Per-note/per-payload sizing knobs. Values differ by task shape.

    binary_nepc makes one LLM call per patient and wants tight context windows
    around each trigger with effectively no per-note cap (the per-patient
    payload cap is the only real ceiling). The longitudinal tasks (stage,
    gleason, longitudinal_NEPC) pack many notes into each call, so they need
    wider per-trigger context to date events but a real per-note cap and a
    smaller per-chunk payload budget.
    """

    context_chars: int
    max_chars: int
    payload_max_chars: int


SNIPPET_PROFILES = {
    # Per-note snippet cap set to the payload budget so note-level truncation is
    # effectively off: a single dense note is never silently truncated mid-signal.
    # Payload cap: 128k-token models at ~4 chars/token give ~500k input chars;
    # budget ~300k for snippets to leave room for the system prompt, JSON
    # scaffolding, and output.
    "binary_nepc": SnippetProfile(context_chars=750, max_chars=300_000, payload_max_chars=300_000),
    # Wider context per trigger for event dating; real per-note cap; chunk at
    # 60k chars (~15k tokens) per LLM call so patients with many notes get
    # multiple calls instead of truncation.
    "longitudinal": SnippetProfile(context_chars=6000, max_chars=30_000, payload_max_chars=60_000),
}
