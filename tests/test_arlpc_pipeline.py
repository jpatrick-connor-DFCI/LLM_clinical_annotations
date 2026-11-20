import json
from pathlib import Path

import polars as pl
import pytest

from preprocessing.cli import collect_arlpc_notes
from tasks.arlpc import run_arlpc_extraction as runner
from tests.test_nepc_pipeline import _collector_args, _runner_args

PATH_TEXT = (
    "SPECIMEN: Liver, core biopsy. CLINICAL HISTORY: prostate cancer on enzalutamide. " + "x " * 200
    + "DIAGNOSIS: Metastatic adenocarcinoma, consistent with prostatic origin. "
    "Tumor cells are negative for AR. NKX3.1 is negative."
)
CLINIC_RESTATED = "Liver biopsy 2019-06 showed AR weak staining. Hormone-refractory since 2018."
CLINIC_POSITIVE = "Prostate biopsy: AR positive (strong, diffuse). PSA stable."
IMAGING_TEXT = "PET: AR negative tumor suspected."


def _write_notes(root):
    pl.DataFrame({
        "RPT_ID": [1], "DFCI_MRN": [101], "EVENT_DATE": ["2021-04-02"],
        "PROC_DESC": ["liver biopsy"], "RPT_TYPE": ["Surgical"],
        "RPT_TEXT": [PATH_TEXT], "FILE": ["pull/path-1.json"],
    }).write_parquet(root / "PATHOLOGY_NOTES.parquet")
    pl.DataFrame({
        "RPT_ID": [1], "DFCI_MRN": [202], "EVENT_DATE": ["2020-01-01"],
        "PROC_DESC": ["PET"], "RPT_TYPE": ["Imaging"],
        "RPT_TEXT": [IMAGING_TEXT], "FILE": ["pull/img-1.json"],
    }).write_parquet(root / "IMAGING_NOTES.parquet")
    pl.DataFrame({
        "RPT_ID": [1, 2],
        "DFCI_MRN": [101, 202],
        "EVENT_DATE": ["2022-01-05", "2020-02-02"],
        "INP_RPT_TYPE": ["Progress"] * 2,
        "PROVIDER_TYPE": ["MD"] * 2,
        "ENCOUNTER_TYPE_DESC": ["Clinic"] * 2,
        "RPT_TEXT": [CLINIC_RESTATED, CLINIC_POSITIVE],
        "FILE": ["pull/progress-1.json", "pull/progress-2.json"],
    }).write_parquet(root / "PROGRESS_NOTES.parquet")


class FakeProvider:
    default_model = "model-a"

    def __init__(self, fail_texts=()):
        self.calls = []
        self.fail_texts = set(fail_texts)

    def build_client(self):
        return object()

    def call_with_retry(self, client, model, messages, max_retries):
        payload = json.loads(messages[1]["content"])
        text = payload["note_text"]
        self.calls.append(payload)
        if any(t in text for t in self.fail_texts):
            return None, "transport_error"
        findings = []
        if "negative for AR" in text:
            findings += [
                {"evidence_type": "ar_ihc", "ar_result": "negative", "favored_origin": "prostate",
                 "specimen_site": "liver biopsy", "stated_date": None, "confidence": "high",
                 "quote": "Tumor cells are negative for AR."},
                {"evidence_type": "ar_target_ihc", "marker": "NKX3.1", "marker_result": "negative",
                 "stated_date": None, "confidence": "high", "quote": "NKX3.1 is negative."},
            ]
        if "AR weak" in text:
            findings += [
                {"evidence_type": "ar_ihc", "ar_result": "reduced", "grade_text": "weak",
                 "stated_date": "2019-06", "confidence": "high",
                 "quote": "Liver biopsy 2019-06 showed AR weak staining."},
                {"evidence_type": "androgen_independent_statement", "stated_date": "2018",
                 "confidence": "high", "quote": "Hormone-refractory since 2018."},
            ]
        if "AR positive" in text:
            findings.append({"evidence_type": "ar_ihc", "ar_result": "positive",
                             "stated_date": None, "confidence": "high",
                             "quote": "AR positive (strong, diffuse)."})
        return json.dumps({"findings": findings}), None


@pytest.fixture
def collected(tmp_path):
    _write_notes(tmp_path)
    args = _collector_args(tmp_path, notes_parquet=[
        tmp_path / "PATHOLOGY_NOTES.parquet",
        tmp_path / "IMAGING_NOTES.parquet",
        tmp_path / "PROGRESS_NOTES.parquet",
    ])
    collect_arlpc_notes.run(args)
    return tmp_path


def test_collector_skips_imaging_and_keeps_whole_pathology(collected):
    evidence = pl.read_parquet(collected / "out" / "arlpc_evidence.parquet")
    assert set(evidence["note_type"]) == {"Pathology", "Clinician"}
    pathology = evidence.filter(pl.col("note_type") == "Pathology")["snippet"].item()
    assert pathology.startswith("SPECIMEN: Liver") and "NKX3.1" in pathology


def test_end_to_end_tiers_dates_and_audit(collected, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(collected))
    assert len(provider.calls) == 3
    by_mrn = {r["DFCI_MRN"]: r for r in labels.iter_rows(named=True)}

    ar_low = by_mrn[101]
    assert ar_low["arlpc_review_status"] == "ok"
    assert ar_low["arlpc_strict"] is True
    # The clinician restates a 2019-06 biopsy: earlier than the 2021 report.
    assert ar_low["arlpc_strict_date"] == "2019-06-01"
    assert ar_low["arlpc_strict_date_precision"] == "month"
    assert ar_low["arlpc_tissue_date"] == "2019-06-01"
    # Broad adds hormone-refractory wording from 2018.
    assert ar_low["arlpc_broad"] is True
    assert ar_low["arlpc_broad_date"] == "2018-01-01"
    assert ar_low["arlpc_broad_evidence_type"] == "androgen_independent_statement"
    assert ar_low["arlpc_has_ar_target_loss"] is True
    assert ar_low["arlpc_ar_target_markers"] == ["nkx3_1"]

    positive = by_mrn[202]
    assert positive["arlpc_strict"] is False and positive["arlpc_broad"] is False
    assert positive["arlpc_n_ar_positive_ihc"] == 1
    assert positive["arlpc_evidence_types"] == ["ar_ihc_positive"]
    assert positive["arlpc_has_ar_target_loss"] is False

    assert by_mrn[303]["arlpc_review_status"] == "no_trigger_notes"
    assert by_mrn[303]["arlpc_strict"] is False


def test_target_loss_alone_sets_no_tier(collected, monkeypatch):
    class TargetOnly(FakeProvider):
        def call_with_retry(self, client, model, messages, max_retries):
            text = json.loads(messages[1]["content"])["note_text"]
            findings = []
            if "NKX3.1 is negative" in text:
                findings.append({"evidence_type": "ar_target_ihc", "marker": "nkx3_1",
                                 "marker_result": "negative", "stated_date": None,
                                 "confidence": "high", "quote": "NKX3.1 is negative."})
            return json.dumps({"findings": findings}), None

    provider = TargetOnly()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    row = runner.run(_runner_args(collected)).filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["arlpc_strict"] is False and row["arlpc_broad"] is False
    assert row["arlpc_has_ar_target_loss"] is True
    assert row["arlpc_strict_date"] is None


def test_resume_retry_rebuild_and_overwrite(collected, monkeypatch):
    failing = FakeProvider(fail_texts={"AR weak"})
    monkeypatch.setattr(runner, "get_provider", lambda name: failing)
    labels = runner.run(_runner_args(collected))
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["arlpc_review_status"] == "partial:1/2"
    assert row["arlpc_strict_date"] == "2021-04-02"

    again = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: again)
    runner.run(_runner_args(collected, retry_failures=True))
    assert len(again.calls) == 1
    labels = pl.read_parquet(collected / "out" / "arlpc_labels.parquet")
    assert labels.filter(pl.col("DFCI_MRN") == 101)["arlpc_strict_date"].item() == "2019-06-01"

    def no_provider(name):
        raise AssertionError("rebuild must not call the provider")

    monkeypatch.setattr(runner, "get_provider", no_provider)
    assert runner.run(_runner_args(collected, rebuild_labels_only=True)).equals(labels)
    meta = pl.read_parquet(collected / "out" / "arlpc_labels.meta.parquet")
    assert meta["ar_gate_version"].item() == runner.SPEC.label_versions["ar_gate_version"]

    monkeypatch.setattr(runner, "get_provider", lambda name: again)
    with pytest.raises(ValueError, match="--overwrite"):
        runner.run(_runner_args(collected, model="model-b"))


def test_pilot_leaves_unread_patients_unlabeled(collected, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(collected, mrns="202"))
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["arlpc_review_status"] == "not_run"
    assert row["arlpc_strict"] is None and row["arlpc_has_ar_target_loss"] is None


@pytest.mark.parametrize("finding, reason", [
    ({"evidence_type": "vibes"}, "invalid_evidence_type:vibes"),
    ({"evidence_type": "ar_ihc", "ar_percent": "150%"}, "invalid_ar_percent"),
    ({"evidence_type": "ar_ihc", "ar_h_score": "high"}, "invalid_ar_h_score"),
])
def test_validate_finding_rejections(finding, reason):
    assert runner.validate_finding(finding, {}) == (None, reason)


def test_validate_finding_coerces():
    fields, _ = runner.validate_finding({
        "evidence_type": "AR_IHC", "ar_result": "Reduced", "ar_percent": "10%",
        "favored_origin": "somewhere", "marker": "psa",
    }, {})
    assert fields["ar_result"] == "reduced" and fields["ar_percent"] == 10.0
    assert fields["favored_origin"] is None and fields["marker"] is None
    fields, _ = runner.validate_finding({"evidence_type": "ar_target_ihc", "marker": "NKX3.1"}, {})
    assert fields["marker"] == "nkx3_1"


def test_collector_captures_ar_results_and_genomic_word_order(tmp_path):
    # No other trigger in these reports can hide a missing AR pattern.
    texts = [
        "Tumor cells are negative for AR.",
        "AR is negative.",
        "AR shows weak staining.",
        "AR is strongly positive.",
        "ctDNA shows AR copy number loss.",
        "NGS shows biallelic deletion of AR.",
        "NGS shows AR biallelic deletion.",
        "NGS shows AR gene copy-number loss.",
    ]
    path = tmp_path / "PATHOLOGY_NOTES.parquet"
    pl.DataFrame({
        "RPT_ID": list(range(len(texts))), "DFCI_MRN": [101] * len(texts),
        "EVENT_DATE": ["2021-04-02"] * len(texts),
        "PROC_DESC": ["biopsy"] * len(texts), "RPT_TYPE": ["Surgical"] * len(texts),
        "RPT_TEXT": texts, "FILE": [f"pull/{i}.json" for i in range(len(texts))],
    }).write_parquet(path)
    collect_arlpc_notes.run(_collector_args(tmp_path, notes_parquet=[path]))
    evidence = pl.read_parquet(tmp_path / "out" / "arlpc_evidence.parquet")
    assert set(evidence["snippet"]) == set(texts)


def test_review_notebook_writes_sample_from_runner_outputs(collected, monkeypatch):
    monkeypatch.setattr(runner, "get_provider", lambda name: FakeProvider())
    runner.run(_runner_args(collected))
    output = collected / "out"
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "prostate_subtype" / "arlpc.ipynb"
    notebook = json.loads(notebook_path.read_text())
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if "REVIEW_STRATA =" in "".join(c["source"]))
    exec(cell, {
        "pl": pl, "Path": Path, "OUTPUT_DIR": output,
        "labels_path": output / "arlpc_labels.parquet",
        "EVIDENCE_PATH": output / "arlpc_evidence.parquet",
        "RUN_REVIEW_SAMPLE": True, "REVIEW_SEED": 1,
    })
    sample = pl.read_parquet(output / "arlpc_review_sample.parquet")
    assert sample.height == 4
    assert set(sample["DFCI_MRN"]) == {101}
    assert set(sample["review_stratum"]) == {"strict_positive"}
    assert sample["snippet"].null_count() == 0
    assert sample["source_note_date"].to_list() == sorted(sample["source_note_date"])
