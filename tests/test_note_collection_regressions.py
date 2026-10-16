import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli import collect_avpc_notes, collect_nepc_dx_notes
from preprocessing.longitudinal import (
    evidence_scan_config_key,
    read_scan_config_meta,
    write_scan_config_meta,
)
from preprocessing.snippets import _scan_note_row
from preprocessing.triggers import NEPC_DX_TRIGGER_REGEX
from tasks import note_extraction
from tasks.avpc import run_avpc_extraction
from tasks.nepc import run_nepc_extraction
from tasks import trial_guard
from tasks.trial_guard import TrialContext


def _write_pathology(root, texts):
    path = root / "PATHOLOGY_NOTES.parquet"
    pl.DataFrame({
        "RPT_ID": list(range(1, len(texts) + 1)),
        "DFCI_MRN": [101 * (i + 1) for i in range(len(texts))],
        "EVENT_DATE": [f"2021-01-{i + 1:02}" for i in range(len(texts))],
        "PROC_DESC": ["prostate biopsy"] * len(texts),
        "RPT_TYPE": ["Surgical"] * len(texts),
        "RPT_TEXT": texts,
        "FILE": [f"path-{i}.json" for i in range(len(texts))],
    }).write_parquet(path)
    return path


def _collector_args(root, source, **overrides):
    args = dict(
        output_dir=root / "out", overwrite=False, mrns="101,202,303",
        mrn_file=None, notes_parquet=[source], note_bundle_path=None,
        note_types=None, context_chars=1500, scan_workers=1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("marker", ["NSE", "INSM-1", "CD 56"])
def test_nepc_collector_includes_supported_marker_spellings(tmp_path, marker):
    text = f"Prostate biopsy: tumor cells are positive for {marker}."
    source = _write_pathology(tmp_path, [text])
    path = collect_nepc_dx_notes.run(_collector_args(tmp_path, source))
    evidence = pl.read_parquet(path)
    assert evidence.height == 1
    assert evidence["snippet"].item() == text


@pytest.mark.parametrize("term", ["neuro endocrine carcinoma", "tNEPC"])
def test_nepc_collector_includes_supported_diagnosis_spellings(tmp_path, term):
    text = f"Prostate biopsy: {term}."
    source = _write_pathology(tmp_path, [text])
    path = collect_nepc_dx_notes.run(_collector_args(tmp_path, source))
    assert pl.read_parquet(path)["snippet"].to_list() == [text]


def test_long_pathology_preserves_header_and_late_addendum(tmp_path):
    header = "SPECIMEN: Prostate biopsy."
    diagnosis = "ADDENDUM: Small cell carcinoma, positive for synaptophysin."
    text = header + "\n" + "Benign tissue details.\n" * 2000 + diagnosis
    assert len(text) > collect_nepc_dx_notes.SNIPPET_MAX_CHARS
    source = _write_pathology(tmp_path, [text])
    path = collect_nepc_dx_notes.run(_collector_args(tmp_path, source))
    snippet = pl.read_parquet(path)["snippet"].item()
    assert snippet.startswith(header)
    assert diagnosis in snippet
    assert len(snippet) <= collect_nepc_dx_notes.SNIPPET_MAX_CHARS


def test_long_pathology_shrinks_context_without_losing_later_triggers():
    header = "SPECIMEN: Prostate biopsy."
    findings = [f"Panel {i}: positive for synaptophysin." for i in range(30)]
    text = header + "\n" + ("Benign tissue details.\n" * 100).join(findings)
    result = _scan_note_row(
        {"DFCI_MRN": 101, "NOTE_TYPE": "Pathology", "EVENT_DATE": "2021-01-01",
         "CLINICAL_TEXT": text},
        context_chars=1500, snippet_max_chars=30_000,
        trigger_regex=NEPC_DX_TRIGGER_REGEX, context_overrides={"Pathology": None},
    )
    assert result["snippet"].startswith(header)
    assert all(finding in result["snippet"] for finding in findings)
    assert len(result["snippet"]) <= 30_000


def test_nepc_collector_rejects_old_evidence_version(tmp_path):
    source = _write_pathology(tmp_path, ["Prostate biopsy: small cell carcinoma."])
    args = _collector_args(tmp_path, source)
    collect_nepc_dx_notes.run(args)
    meta_path = args.output_dir / "nepc_evidence.meta.parquet"
    metadata = read_scan_config_meta(meta_path)
    scan_config = metadata.pop("scan_config")
    metadata["evidence_schema_version"] = "nepc-dx-note-evidence-parquet-v1"
    write_scan_config_meta(meta_path, scan_config, **metadata)
    with pytest.raises(ValueError, match="predates.*contract"):
        collect_nepc_dx_notes.run(args)


@pytest.mark.parametrize("prefix,collector,runner", [
    ("avpc", collect_avpc_notes, run_avpc_extraction),
    ("nepc", collect_nepc_dx_notes, run_nepc_extraction),
])
@pytest.mark.parametrize("pilot", [[], [202]])
def test_notebook_pilot_limits_calls_but_preserves_cohort(
    tmp_path, monkeypatch, prefix, collector, runner, pilot,
):
    source = _write_pathology(tmp_path, [
        "Prostate biopsy: small cell carcinoma, specimen A.",
        "Prostate biopsy: small cell carcinoma, specimen B.",
    ])
    cohort = tmp_path / "cohort.csv"
    pl.DataFrame({"DFCI_MRN": [101, 202, 303]}).write_csv(cohort)
    collector.run(_collector_args(tmp_path, source, mrn_file=cohort))
    namespace = dict(
        PYTHON="python", PROVIDER="dfci_gpt", OUTPUT_DIR=tmp_path / "out",
        EVIDENCE_PATH=tmp_path / "out" / f"{prefix}_evidence.parquet",
        MODEL_NAME="fake-model", MAX_WORKERS=1, MAX_RETRIES=1,
        MRNS=pilot, MRN_FILE=cohort, LIMIT_NOTES=None,
        REBUILD_LABELS_ONLY=False, OVERWRITE=False,
        shell_join=lambda parts: " ".join(map(str, parts)),
    )
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / f"{prefix}.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    command_cell = next(
        "".join(cell["source"]) for cell in cells
        if "run_extraction_cmd = [" in "".join(cell["source"])
    )
    exec(compile(command_cell, str(notebook), "exec"), namespace)
    args = note_extraction.build_parser(runner.SPEC).parse_args(
        list(map(str, namespace["run_extraction_cmd"][2:]))
    )

    class FakeProvider:
        default_model = "fake-model"

        def __init__(self):
            self.calls = []

        def build_client(self):
            return None

        def call_with_retry(self, client, model, messages, max_retries):
            self.calls.append(json.loads(messages[1]["content"])["note_text"])
            return '{"findings": []}', None

    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(args)
    assert len(provider.calls) == (1 if pilot else 2)
    if pilot:
        assert "specimen B" in provider.calls[0]
    assert set(labels["DFCI_MRN"]) == {101, 202, 303}
    if pilot:
        assert labels.filter(pl.col("DFCI_MRN") == 101)["review_status"].item() == "not_run"


def _write_clinician(root, texts):
    path = root / "PROGRESS_NOTES.parquet"
    pl.DataFrame({
        "RPT_ID": list(range(1, len(texts) + 1)),
        "DFCI_MRN": [101 * (i + 1) for i in range(len(texts))],
        "EVENT_DATE": [f"2021-01-{i + 1:02}" for i in range(len(texts))],
        "INP_RPT_TYPE": ["Progress"] * len(texts),
        "PROVIDER_TYPE": ["MD"] * len(texts),
        "ENCOUNTER_TYPE_DESC": ["Clinic"] * len(texts),
        "RPT_TEXT": texts,
        "FILE": [f"clinic-{i}.json" for i in range(len(texts))],
    }).write_parquet(path)
    return path


@pytest.mark.parametrize("collector,quote", [
    (collect_nepc_dx_notes, "Small cell carcinoma of the prostate."),
    (collect_avpc_notes, "Aggressive-variant prostate cancer."),
])
@pytest.mark.parametrize("workers", [1, 2])
def test_collectors_keep_distant_eligibility_headers_and_clinical_resets(
    tmp_path, collector, quote, workers,
):
    padding = "General background and administrative details.\n" * 120
    assert len(padding) > 3000
    trial_only = (
        "INCLUSION CRITERIA\n" + padding + "1. " + quote + "\n"
        + padding + "Assessment:\nRoutine follow-up.\n"
    )
    clinical_too = trial_only + padding + quote
    source = _write_clinician(tmp_path, [trial_only, clinical_too])
    path = collector.run(_collector_args(tmp_path, source, scan_workers=workers))
    snippets = dict(pl.read_parquet(path).select("DFCI_MRN", "snippet").iter_rows())
    for snippet in snippets.values():
        assert "INCLUSION CRITERIA" in snippet
        assert "Assessment:" in snippet
        assert quote in snippet
    assert TrialContext(snippets[101]).reason(quote) == "boilerplate:eligibility_section"
    assert TrialContext(snippets[202]).reason(quote) is None


@pytest.mark.parametrize("context_chars", [0, 1500])
def test_bounded_snippets_preserve_trial_anchors_and_actual_trigger_counts(context_chars):
    padding = "General background and administrative details.\n" * 100
    trial_only = "INCLUSION CRITERIA\n" + padding + "small cell\n"
    text = trial_only + padding + "Assessment:\n" + padding + "small cell"
    kwargs = dict(
        context_chars=context_chars, snippet_max_chars=200,
        trigger_regex=NEPC_DX_TRIGGER_REGEX, preserve_trial_context=True,
    )
    candidate = _scan_note_row(
        {"DFCI_MRN": 101, "NOTE_TYPE": "Clinician", "CLINICAL_TEXT": text}, **kwargs,
    )
    snippet = candidate["snippet"]
    assert len(snippet) <= 200
    assert "INCLUSION CRITERIA" in snippet and "Assessment:" in snippet
    assert snippet.count("small cell") == 2
    assert candidate["trigger_categories"] == ["nepc_dx"]
    assert candidate["trigger_count"] == 2
    assert TrialContext(snippet).reason("small cell") is None
    trial_candidate = _scan_note_row(
        {"DFCI_MRN": 101, "NOTE_TYPE": "Clinician", "CLINICAL_TEXT": trial_only}, **kwargs,
    )
    assert TrialContext(trial_candidate["snippet"]).reason("small cell") == (
        "boilerplate:eligibility_section"
    )


def test_trial_context_preservation_is_opt_in_for_legacy_scanning():
    padding = "General background and administrative details.\n" * 120
    text = "INCLUSION CRITERIA\n" + padding + "Small cell carcinoma of the prostate."
    row = {"DFCI_MRN": 101, "NOTE_TYPE": "Clinician", "CLINICAL_TEXT": text}
    kwargs = dict(context_chars=100, snippet_max_chars=30_000,
                  trigger_regex=NEPC_DX_TRIGGER_REGEX)
    legacy = _scan_note_row(row, **kwargs)
    protected = _scan_note_row(row, preserve_trial_context=True, **kwargs)
    assert "INCLUSION CRITERIA" not in legacy["snippet"]
    assert "INCLUSION CRITERIA" in protected["snippet"]
    row["CLINICAL_TEXT"] = "INCLUSION CRITERIA\nAge at least 18.\nAssessment:\nRoutine visit."
    assert _scan_note_row(row, preserve_trial_context=True, **kwargs) is None


def test_trial_guard_version_invalidates_only_enabled_scan_fingerprints(monkeypatch):
    notes = pl.DataFrame({"DFCI_MRN": [101], "CLINICAL_TEXT": ["small cell"]})
    kwargs = dict(context_chars=1500, snippet_max_chars=30_000, payload_max_chars=30_000)
    legacy = evidence_scan_config_key(notes, NEPC_DX_TRIGGER_REGEX, **kwargs)
    protected = evidence_scan_config_key(
        notes, NEPC_DX_TRIGGER_REGEX, preserve_trial_context=True, **kwargs,
    )
    assert protected != legacy
    assert evidence_scan_config_key(
        notes, NEPC_DX_TRIGGER_REGEX, preserve_trial_context=False, **kwargs,
    ) == legacy
    monkeypatch.setattr(trial_guard, "TRIAL_GUARD_VERSION", "changed-guard-version")
    assert evidence_scan_config_key(notes, NEPC_DX_TRIGGER_REGEX, **kwargs) == legacy
    assert evidence_scan_config_key(
        notes, NEPC_DX_TRIGGER_REGEX, preserve_trial_context=True, **kwargs,
    ) != protected
