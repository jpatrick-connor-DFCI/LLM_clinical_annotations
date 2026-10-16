import json
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli import collect_avpc_notes
from tasks.avpc import run_avpc_extraction as runner
from tasks.avpc.criteria import CRITERIA

IMAGING = "Bone scan: innumerable osseous metastases, predominantly lytic osseous metastases."
CLINIC = (
    "Metastatic castration-resistant prostate cancer. At symptomatic progression "
    "PSA was 3.1 ng/mL. Started leuprolide in 2020-01."
)


def _write_notes(root):
    pl.DataFrame({
        "RPT_ID": [1],
        "DFCI_MRN": [101],
        "EVENT_DATE": ["2020-06-10"],
        "PROC_DESC": ["bone scan"],
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
        "RPT_TEXT": [CLINIC, "Hormone-sensitive disease, no metastases, on ADT."],
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
        if "innumerable" in text:
            findings.append({
                "item": "bone_metastasis", "lytic_predominant": True, "count": None,
                "high_volume_descriptor": "innumerable", "stated_date": None,
                "quote": "innumerable osseous metastases, predominantly lytic osseous metastases",
                "confidence": "high",
            })
        if "symptomatic progression" in text:
            findings += [
                {"item": "castration_resistance", "stated_date": None, "confidence": "high",
                 "quote": "Metastatic castration-resistant prostate cancer."},
                {"item": "low_psa", "psa_ng_ml": "3.1",
                 "psa_context": "symptomatic_crpc_progression", "stated_date": None,
                 "quote": "At symptomatic progression PSA was 3.1 ng/mL.", "confidence": "high"},
                {"item": "adt_start", "agent": "leuprolide", "stated_date": "2020-01",
                 "quote": "Started leuprolide in 2020-01.", "confidence": "high"},
                {"item": "bulky_lymphadenopathy", "size_cm": "a lot",
                 "quote": "At symptomatic progression", "confidence": "low"},
            ]
        return json.dumps({"findings": findings}), None


@pytest.fixture
def out_dir(tmp_path):
    _write_notes(tmp_path)
    collect_avpc_notes.run(SimpleNamespace(
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


def test_end_to_end_avpc_labels(out_dir, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(out_dir))
    assert provider.calls == 3

    expected = {"DFCI_MRN", "is_metastatic", "is_castration_resistant", "is_mcrpc",
                "avpc_criteria", "is_avpc", "avpc_date", "c7_interval_days"}
    for name in CRITERIA:
        key = name.lower()
        expected |= {f"{key}_met", f"{key}_quote", f"{key}_date", f"{key}_source_note_date"}
    assert expected <= set(labels.columns)

    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["is_metastatic"] and row["is_castration_resistant"] and row["is_mcrpc"]
    assert row["avpc_criteria"] == ["C3", "C5", "C7"]
    assert row["c3_quote"].startswith("innumerable osseous metastases")
    assert row["c5_psa_quote"] == "At symptomatic progression PSA was 3.1 ng/mL."
    # ADT stated 2020-01, CRPC first documented 2020-06-20.
    assert row["c7_interval_days"] == 171
    assert row["is_avpc"] is True
    assert row["avpc_date"] == "2020-06-20"

    other = labels.filter(pl.col("DFCI_MRN") == 202).row(0, named=True)
    assert other["is_avpc"] is False and other["review_status"] == "ok"

    rejected = pl.read_parquet(out_dir / "avpc_rejected_findings.parquet")
    assert rejected["reason"].to_list() == ["invalid_size_cm"]
    findings = pl.read_parquet(out_dir / "avpc_findings.parquet")
    psa = findings.filter(pl.col("item") == "low_psa")
    assert psa["psa_ng_ml"].item() == pytest.approx(3.1)
    adt = findings.filter(pl.col("item") == "adt_start").row(0, named=True)
    assert adt["date_precision"] == "month" and adt["date_source"] == "stated"


def test_rebuild_labels_only_makes_no_calls(out_dir, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    first = runner.run(_runner_args(out_dir))

    def no_provider(name):
        raise AssertionError("rebuild must not call the provider")

    monkeypatch.setattr(runner, "get_provider", no_provider)
    assert runner.run(_runner_args(out_dir, rebuild_labels_only=True)).equals(first)
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.run(_runner_args(out_dir, rebuild_labels_only=True, overwrite=True))


def test_validate_finding_coerces_types():
    fields, reason = runner.validate_finding(
        {"item": "Bone_Metastasis", "count": "24", "lytic_predominant": "true"}, {}
    )
    assert reason is None
    assert fields["item"] == "bone_metastasis"
    assert fields["count"] == 24 and fields["lytic_predominant"] is True
    assert runner.validate_finding({"item": "bone_metastasis", "count": 2.5}, {}) == (
        None, "invalid_count"
    )
    fields, _ = runner.validate_finding({"item": "low_psa", "psa_context": "whenever"}, {})
    assert fields["psa_context"] is None
    assert runner.validate_finding({"item": "nope"}, {})[1] == "invalid_item:nope"
