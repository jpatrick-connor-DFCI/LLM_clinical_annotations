import json
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli import collect_avpc_adjacent_notes
from tasks.avpc_adjacent import run_avpc_adjacent_extraction as runner

IMAGING = "CT: diffuse osseous metastases with mixed lytic and sclerotic lesions. Hepatic metastases."
CLINIC = (
    "Hormone-sensitive prostate cancer. At initial diagnosis PSA was 4.2 ng/mL. "
    "Picture consistent with aggressive variant prostate cancer."
)


def _write_notes(root):
    pl.DataFrame({
        "RPT_ID": [1],
        "DFCI_MRN": [101],
        "EVENT_DATE": ["2020-06-10"],
        "PROC_DESC": ["CT chest abdomen pelvis"],
        "RPT_TYPE": ["Imaging"],
        "RPT_TEXT": [IMAGING],
        "FILE": ["pull/image-1.json"],
    }).write_parquet(root / "IMAGING_NOTES.parquet")
    pl.DataFrame({
        "RPT_ID": [1, 2],
        "DFCI_MRN": [101, 202],
        "EVENT_DATE": ["2020-06-20", "2020-06-20"],
        "INP_RPT_TYPE": ["Progress"] * 2,
        "PROVIDER_TYPE": ["MD"] * 2,
        "ENCOUNTER_TYPE_DESC": ["Clinic"] * 2,
        "RPT_TEXT": [CLINIC, "Localized prostate cancer, no metastases, PSA 5.0."],
        "FILE": ["pull/progress-1.json", "pull/progress-2.json"],
    }).write_parquet(root / "PROGRESS_NOTES.parquet")


class FakeProvider:
    default_model = "model-a"

    def __init__(self):
        self.calls = 0

    def build_client(self):
        return object()

    def call_with_retry(self, client, model, messages, max_retries):
        self.calls += 1
        text = json.loads(messages[1]["content"])["note_text"]
        findings = []
        if "diffuse osseous" in text:
            findings += [
                {"item": "bone_metastasis", "lytic_component": "true", "lytic_predominant": False,
                 "count": None, "high_volume_descriptor": "diffuse", "stated_date": None,
                 "quote": "diffuse osseous metastases with mixed lytic and sclerotic lesions",
                 "confidence": "high"},
                {"item": "visceral_metastasis", "organ": "liver", "exclusively_visceral": False,
                 "stated_date": None, "quote": "Hepatic metastases.", "confidence": "high"},
            ]
        if "aggressive variant" in text:
            findings += [
                {"item": "low_psa", "psa_ng_ml": 4.2, "psa_context": "initial_presentation_pre_adt",
                 "stated_date": "2020-06", "quote": "At initial diagnosis PSA was 4.2 ng/mL.",
                 "confidence": "high"},
                {"item": "aggressive_variant_statement", "term": "aggressive variant",
                 "stated_date": None, "confidence": "medium",
                 "quote": "Picture consistent with aggressive variant prostate cancer."},
            ]
        return json.dumps({"findings": findings}), None


@pytest.fixture
def out_dir(tmp_path):
    _write_notes(tmp_path)
    collect_avpc_adjacent_notes.run(SimpleNamespace(
        output_dir=tmp_path / "out", overwrite=False, mrns="101,202", mrn_file=None,
        notes_parquet=[tmp_path / "IMAGING_NOTES.parquet", tmp_path / "PROGRESS_NOTES.parquet"],
        note_bundle_path=None, note_types=None, context_chars=2000, scan_workers=1,
    ))
    return tmp_path / "out"


def _runner_args(out_dir, **overrides):
    args = dict(
        output_dir=out_dir, evidence_path=None, mrns=None, mrn_file=None,
        provider="dfci_gpt", model=None, max_workers=2, max_retries=1, limit_notes=None,
        overwrite=False, retry_failures=False, rebuild_labels_only=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_end_to_end_avpc_adjacent_labels(out_dir, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(out_dir))
    assert provider.calls == 3
    assert set(labels.columns) == set(runner.LABEL_SCHEMA)
    # Merge-facing labels and onset precision are unprefixed.
    assert {c for c in labels.columns if not c.startswith("avpc_adj_")} == {
        "DFCI_MRN", "is_avpc_adjacent", "avpc_adjacent_date", "avpc_adjacent_date_precision"}

    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["avpc_adj_criteria"] == ["A1", "A2", "A3", "A4"]
    assert row["is_avpc_adjacent"] is True
    # No CRPC documented: not strict AVPC, and the adjacent label needs no state.
    assert row["avpc_adj_is_castration_resistant"] is False
    assert row["avpc_adj_is_avpc_strict"] is False
    assert row["avpc_adj_a3_psa_quote"] == "At initial diagnosis PSA was 4.2 ng/mL."
    assert row["avpc_adjacent_date"] == "2020-06-10"

    other = labels.filter(pl.col("DFCI_MRN") == 202).row(0, named=True)
    assert other["is_avpc_adjacent"] is False and other["avpc_adj_review_status"] == "ok"

    findings = pl.read_parquet(out_dir / "avpc_adjacent_findings.parquet")
    lytic = findings.filter(pl.col("item") == "bone_metastasis")
    assert lytic["lytic_component"].item() is True
    meta = pl.read_parquet(out_dir / "avpc_adjacent_labels.meta.parquet")
    assert "adjacent_criteria_version" in meta.columns


def test_rebuild_labels_only_makes_no_calls(out_dir, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    first = runner.run(_runner_args(out_dir))
    monkeypatch.setattr(runner, "get_provider", lambda name: pytest.fail("no calls expected"))
    assert runner.run(_runner_args(out_dir, rebuild_labels_only=True)).equals(first)


def test_unknown_item_rejected():
    fields, reason = runner.validate_finding({"item": "aggressive_disease"}, {})
    assert fields is None and reason == "invalid_item:aggressive_disease"
    fields, reason = runner.validate_finding(
        {"item": "bone_metastasis", "lytic_component": "maybe"}, {})
    assert reason == "invalid_lytic_component"
