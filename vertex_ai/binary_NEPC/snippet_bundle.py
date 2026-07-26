"""Persistence for the binary NEPC patient-snippet pipeline artifact."""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path


SNIPPET_BUNDLE_FILENAME = "LLM_NEPC_classifier_patient_snippets.json.gz"
SNIPPET_BUNDLE_FORMAT = "binary_nepc_patient_snippets"
SNIPPET_BUNDLE_VERSION = 1
REQUIRED_SNIPPET_FIELDS = {
    "note_date",
    "note_type",
    "trigger_categories",
    "snippet",
}


def _normalize_snippets(patient_snippets):
    normalized = {}
    for raw_mrn, snippets in patient_snippets.items():
        mrn = int(raw_mrn)
        if not isinstance(snippets, list):
            raise ValueError(f"Snippets for MRN {mrn} must be a list")
        normalized_snippets = []
        for index, snippet in enumerate(snippets):
            if not isinstance(snippet, dict):
                raise ValueError(f"Snippet {index} for MRN {mrn} must be an object")
            missing = REQUIRED_SNIPPET_FIELDS - set(snippet)
            if missing:
                raise ValueError(
                    f"Snippet {index} for MRN {mrn} is missing: {sorted(missing)}"
                )
            if not isinstance(snippet["trigger_categories"], list):
                raise ValueError(
                    f"trigger_categories for snippet {index}, MRN {mrn} must be a list"
                )
            if not isinstance(snippet["snippet"], str):
                raise ValueError(
                    f"snippet text for snippet {index}, MRN {mrn} must be a string"
                )
            normalized_snippets.append(
                {
                    "note_date": snippet["note_date"],
                    "note_type": snippet["note_type"],
                    "trigger_categories": list(snippet["trigger_categories"]),
                    "snippet": snippet["snippet"],
                }
            )
        normalized[mrn] = normalized_snippets
    return normalized


def write_snippet_bundle(path, *, all_mrns, patient_snippets, metadata=None):
    """Atomically write all cohort MRNs and triggered patient snippets."""
    path = Path(path)
    all_mrns = sorted({int(mrn) for mrn in all_mrns})
    normalized = _normalize_snippets(patient_snippets)
    unexpected = set(normalized) - set(all_mrns)
    if unexpected:
        raise ValueError(
            f"Snippet patients absent from all_mrns: {sorted(unexpected)[:10]}"
        )

    payload = {
        "format": SNIPPET_BUNDLE_FORMAT,
        "version": SNIPPET_BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "all_mrns": all_mrns,
        "patient_snippets": {
            str(mrn): normalized[mrn] for mrn in sorted(normalized)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary_path.replace(path)


def load_snippet_bundle(path, *, selected_mrns=None):
    """Load and validate a snippet artifact, optionally filtering its cohort."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Patient snippet bundle not found: {path}. "
            "Run binary_NEPC/compile_patient_snippets.py first."
        )
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid patient snippet bundle: {path}") from error

    if payload.get("format") != SNIPPET_BUNDLE_FORMAT:
        raise ValueError(f"Unrecognized patient snippet bundle format: {path}")
    if payload.get("version") != SNIPPET_BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported patient snippet bundle version "
            f"{payload.get('version')!r}: {path}"
        )

    all_mrns = {int(mrn) for mrn in payload.get("all_mrns", [])}
    patient_snippets = _normalize_snippets(payload.get("patient_snippets", {}))
    unexpected = set(patient_snippets) - all_mrns
    if unexpected:
        raise ValueError(
            f"Snippet patients absent from bundle cohort: {sorted(unexpected)[:10]}"
        )

    if selected_mrns is not None:
        selected_mrns = {int(mrn) for mrn in selected_mrns}
        all_mrns &= selected_mrns
        patient_snippets = {
            mrn: snippets
            for mrn, snippets in patient_snippets.items()
            if mrn in selected_mrns
        }
    return all_mrns, patient_snippets, payload.get("metadata", {})
