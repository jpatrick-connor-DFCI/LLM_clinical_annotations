"""End to end: course collector -> per-note runner (fake provider) -> BCR,
progression and mCRPC builders over synthetic LABS / MEDICATIONS / registry."""

import json
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli import collect_course_notes
from preprocessing.parquet_io import read_metadata
from tasks.avpc.run_avpc_extraction import SPEC as AVPC_SPEC
from tasks.bcr import build_bcr_labels
from tasks.disease_course import run_course_extraction as runner
from tasks.mcrpc import build_mcrpc_labels
from tasks.nepc.run_nepc_extraction import SPEC as NEPC_SPEC
from tasks.note_extraction import run_fingerprint
from tasks.progression import build_progression_events

SURGERY = "Status post radical prostatectomy on 2018-03-15. PSA undetectable on 2018-05-01."
CLINIC = (
    "Metastatic castration-resistant prostate cancer. PSMA PET on 2020-12-01 showed new "
    "bone lesions consistent with progression. Started cabazitaxel on 2021-02-01."
)
TRIAL = "Eligibility criteria for the study include rising PSA 5.0 ng/mL or higher on prostatectomy follow-up."
FAILING = "FAILME prostatectomy and PSA history."


def _write_notes(root):
    pl.DataFrame({
        "RPT_ID": [1, 2, 3, 4],
        "DFCI_MRN": [101, 101, 101, 202],
        "EVENT_DATE": ["2018-05-01", "2021-02-10", "2021-03-01", "2021-01-01"],
        "INP_RPT_TYPE": ["Progress"] * 4,
        "PROVIDER_TYPE": ["MD"] * 4,
        "ENCOUNTER_TYPE_DESC": ["Clinic"] * 4,
        "RPT_TEXT": [SURGERY, CLINIC, TRIAL, FAILING],
        "FILE": [f"pull/progress-{i}.json" for i in range(4)],
    }).write_parquet(root / "PROGRESS_NOTES.parquet")


def _write_structured(root):
    pl.DataFrame({
        "DFCI_MRN": [101, 101, 101, 202],
        "SPECIMEN_COLLECT_DT": ["2018-06-01", "2019-01-10", "2019-04-10", "2020-01-01"],
        "TEST_TYPE_CD": ["PSA"] * 4,
        "TEST_TYPE_DESCR": ["PSA"] * 4,
        "NUMERIC_RESULT": [9999999.0, 0.25, 0.4, 3.0],
        "TEXT_RESULT": ["<0.1", "0.25", "0.4", "3.0"],
        "RESULT_UOM_NM": ["ng/mL"] * 4,
    }).write_parquet(root / "LABS.parquet")
    pl.DataFrame({
        "DFCI_MRN": [101],
        "NCI_PREFERRED_MED_NM": ["LEUPROLIDE ACETATE"],
        "MED_START_DT": ["2019-06-01"],
    }).write_parquet(root / "MEDICATIONS.parquet")
    pl.DataFrame({
        "DFCI_MRN": [101, 202],
        "HYBRID_DEATH_DT": ["2022-03-01", "2021-06-01"],
        "DERIVED_LAST_ALIVE_DATE": ["2022-02-01", "2021-05-01"],
    }).write_parquet(root / "PT_INFO_STATUS_REGISTRATION.parquet")
    pl.DataFrame({
        "DFCI_MRN": [101, 202],
        "is_metastatic": [True, False],
        "metastatic_date": ["2020-12-01", None],
        "metastatic_date_precision": ["day", None],
    }).write_parquet(root / "avpc_labels.parquet")


class FakeProvider:
    default_model = "model-a"

    def __init__(self):
        self.calls = 0

    def build_client(self):
        return object()

    def call_with_retry(self, client, model, messages, max_retries):
        self.calls += 1
        text = json.loads(messages[1]["content"])["note_text"]
        if "FAILME" in text:
            return None, "provider_error"
        findings = []
        if "Status post radical prostatectomy" in text:
            findings += [
                {"item": "radical_prostatectomy", "stated_date": "2018-03-15", "confidence": "high",
                 "quote": "Status post radical prostatectomy on 2018-03-15."},
                {"item": "psa_value", "psa_ng_ml": None, "value_comparator": "undetectable",
                 "value_timing": "dated", "stated_date": "2018-05-01", "confidence": "high",
                 "quote": "PSA undetectable on 2018-05-01."},
            ]
        if "Started cabazitaxel" in text:
            findings += [
                {"item": "mcrpc_statement", "stated_date": None, "confidence": "high",
                 "quote": "Metastatic castration-resistant prostate cancer."},
                {"item": "progression_event", "progression_type": "radiographic",
                 "stated_date": "2020-12-01", "confidence": "high",
                 "quote": "PSMA PET on 2020-12-01 showed new bone lesions consistent with progression."},
                {"item": "therapy_start", "agent": "cabazitaxel", "stated_date": "2021-02-01",
                 "confidence": "high", "quote": "Started cabazitaxel on 2021-02-01."},
            ]
        if "Eligibility criteria" in text:
            findings.append({
                "item": "psa_value", "psa_ng_ml": 5.0, "value_comparator": "=",
                "value_timing": "current", "stated_date": None, "confidence": "low",
                "quote": "rising PSA 5.0 ng/mL or higher",
            })
        return json.dumps({"findings": findings}), None


def _runner_args(out_dir, **overrides):
    args = dict(
        output_dir=out_dir, evidence_path=None, mrns=None, mrn_file=None,
        provider="dfci_gpt", model=None, max_workers=2, max_retries=1, limit_notes=None,
        overwrite=False, retry_failures=False, rebuild_labels_only=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def course(tmp_path, monkeypatch):
    _write_notes(tmp_path)
    _write_structured(tmp_path)
    out = tmp_path / "out"
    collect_course_notes.run(SimpleNamespace(
        output_dir=out, overwrite=False, mrns="101,202", mrn_file=None,
        notes_parquet=[tmp_path / "PROGRESS_NOTES.parquet"],
        note_bundle_path=None, note_types=None, context_chars=2000, scan_workers=1,
    ))
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(out))
    return SimpleNamespace(root=tmp_path, out=out, labels=labels, provider=provider)


def _builder_args(course, *extra):
    return ["--output-dir", str(course.out), "--profile-data-path", str(course.root), *extra]


def test_course_labels(course):
    assert course.provider.calls == 4
    row = course.labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["review_status"] == "ok"
    assert row["local_therapy"] == "rp" and row["rp_date"] == "2018-03-15"
    assert row["n_psa_values"] == 1
    assert row["gate_rejections"] and row["gate_rejections"][0].startswith("psa_value:")
    failed = course.labels.filter(pl.col("DFCI_MRN") == 202).row(0, named=True)
    assert failed["review_status"] == "failed" and failed["local_therapy"] is None


def test_bcr_builder(course):
    labels = build_bcr_labels.main(_builder_args(course))
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["bcr"] is True and row["bcr_basis"] == "psa_aua"
    assert row["bcr_date"] == "2019-01-10" and row["psa_bcr_confirmation_date"] == "2019-04-10"
    assert row["index_date"] == "2018-03-15"
    # The note's undetectable PSA (2018-05-01) clears before the LABS <0.1.
    assert row["psa_clearance_date"] == "2018-05-01"
    # The cropped trial-eligibility PSA never reaches the PSA rules.
    assert row["n_psa_measurements"] == 4
    failed = labels.filter(pl.col("DFCI_MRN") == 202).row(0, named=True)
    assert failed["review_status"] == "failed" and failed["bcr"] is None
    meta = read_metadata(course.out / "bcr_labels.meta.parquet")
    assert meta["bcr_criteria_version"] == "bcr-criteria-v1"
    assert json.loads(meta["structured_sources"])["labs"]["path"].endswith("LABS.parquet")
    assert json.loads(meta["psa_rule_defaults"])["undetectable_counts_as_clearance"] is True


def test_progression_builder(course):
    events, labels = build_progression_events.main(_builder_args(course))
    patient = events.filter(pl.col("DFCI_MRN") == 101)
    assert patient.select("event_type", "event_date", "basis").rows() == [
        ("radiographic", "2020-12-01", "statement"), ("death", "2022-03-01", "registry"),
    ]
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["first_any_progression_date"] == "2020-12-01"
    assert row["first_radiographic_progression_basis"] == "statement"
    assert row["death_date"] == "2022-03-01" and row["n_progression_events"] == 1
    # A failed review keeps the registry death but no note-derived events.
    failed = labels.filter(pl.col("DFCI_MRN") == 202).row(0, named=True)
    assert failed["death_date"] == "2021-06-01" and failed["first_any_progression_date"] is None
    assert events.filter(pl.col("DFCI_MRN") == 202)["event_type"].to_list() == ["death"]
    assert (course.out / "progression_events.meta.parquet").exists()


def test_mcrpc_builder(course):
    labels = build_mcrpc_labels.main(_builder_args(
        course, "--met-labels", str(course.root / "avpc_labels.parquet")
    ))
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["is_mcrpc"] is True
    assert row["mcrpc_basis"] == "drug_proxy" and row["mcrpc_date"] == "2021-02-01"
    assert row["proxy_precedes_primary"] is True and row["proxy_agent"] == "cabazitaxel"
    assert row["primary_date"] == "2021-02-10" and row["met_date"] == "2020-12-01"
    assert labels.filter(pl.col("DFCI_MRN") == 202)["is_mcrpc"].item() is None
    meta = read_metadata(course.out / "mcrpc_labels.meta.parquet")
    assert meta["met_source"] == "avpc"
    assert json.loads(meta["met_labels"])["sha256"]
    with pytest.raises(FileNotFoundError):
        build_mcrpc_labels.main(_builder_args(
            course, "--met-labels", str(course.root / "missing.parquet")
        ))


def test_rebuild_labels_only_is_stable(course, monkeypatch):
    monkeypatch.setattr(runner, "get_provider", lambda name: pytest.fail("no LLM calls expected"))
    rebuilt = runner.run(_runner_args(course.out, rebuild_labels_only=True))
    assert rebuilt.equals(course.labels)


def test_existing_fingerprints_unchanged():
    assert run_fingerprint(AVPC_SPEC, "e", "p", "m") == "02d4991449698b3eb9de"
    assert run_fingerprint(NEPC_SPEC, "e", "p", "m") == "d0ad7b81d8b85bdf3a37"
