import json
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli import collect_avpc_notes, collect_nepc_dx_notes
from tasks import note_extraction
from tasks.nepc import run_nepc_extraction as runner

PATH_HEADER = "SPECIMEN: Liver, core biopsy. CLINICAL HISTORY: prostate cancer on ADT. " + "x " * 200
PATH_TEXT = (
    PATH_HEADER
    + "DIAGNOSIS: Metastatic small cell carcinoma, consistent with prostatic origin. "
    "Tumor cells are positive for synaptophysin and INSM1."
)
CLINIC_NEGATIVE = "Prostate biopsy reviewed. No small cell component is identified. Plan ADT."
CLINIC_LATER = "Assessment: small cell carcinoma of the prostate, s/p carboplatin/etoposide."


def _write_notes(root):
    pl.DataFrame({
        "RPT_ID": [1],
        "DFCI_MRN": [101],
        "EVENT_DATE": ["2021-04-02"],
        "PROC_DESC": ["liver biopsy"],
        "RPT_TYPE": ["Surgical"],
        "RPT_TEXT": [PATH_TEXT],
        "FILE": ["pull/path-1.json"],
    }).write_parquet(root / "PATHOLOGY_NOTES.parquet")
    pl.DataFrame({
        "RPT_ID": [1, 2, 3],
        "DFCI_MRN": [101, 202, 202],
        "EVENT_DATE": ["2022-01-05", "2020-02-02", "2020-03-03"],
        "INP_RPT_TYPE": ["Progress"] * 3,
        "PROVIDER_TYPE": ["MD"] * 3,
        "ENCOUNTER_TYPE_DESC": ["Clinic"] * 3,
        "RPT_TEXT": [CLINIC_LATER, CLINIC_NEGATIVE, "Routine visit, PSA stable."],
        "FILE": ["pull/progress-1.json", "pull/progress-2.json", "pull/progress-3.json"],
    }).write_parquet(root / "PROGRESS_NOTES.parquet")


def _collector_args(root, **overrides):
    args = dict(
        output_dir=root / "out",
        overwrite=False,
        mrns="101,202,303",
        mrn_file=None,
        notes_parquet=[root / "PATHOLOGY_NOTES.parquet", root / "PROGRESS_NOTES.parquet"],
        note_bundle_path=None,
        note_types=None,
        context_chars=200,
        scan_workers=1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _runner_args(root, **overrides):
    args = dict(
        output_dir=root / "out",
        evidence_path=None,
        mrns=None,
        mrn_file=None,
        provider="dfci_gpt",
        model=None,
        max_workers=2,
        max_retries=1,
        limit_notes=None,
        overwrite=False,
        retry_failures=False,
        rebuild_labels_only=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


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
        if "Metastatic small cell carcinoma" in text:
            findings += [
                {"evidence_type": "nepc_diagnosis", "specimen_site": "liver biopsy",
                 "stated_date": None, "confidence": "high",
                 "quote": "Metastatic small cell carcinoma, consistent with prostatic origin."},
                {"evidence_type": "positive_ne_ihc", "specimen_site": "liver biopsy",
                 "stated_date": None, "confidence": "high",
                 "quote": "Tumor cells are positive for synaptophysin and INSM1."},
                # Invented: not in the note, must be rejected without losing the others.
                {"evidence_type": "nepc_diagnosis", "specimen_site": None,
                 "stated_date": None, "confidence": "low",
                 "quote": "Pure small cell carcinoma of the prostate, extensive."},
                # Stated after its own note.
                {"evidence_type": "positive_ne_ihc", "specimen_site": None,
                 "stated_date": "2023-01", "confidence": "low",
                 "quote": "positive for synaptophysin and INSM1"},
            ]
        if "s/p carboplatin" in text:
            findings.append({
                "evidence_type": "nepc_diagnosis", "specimen_site": None,
                "stated_date": "2020", "confidence": "high",
                "quote": "small cell carcinoma of the prostate, s/p carboplatin/etoposide",
            })
        if "No small cell component" in text:
            # The model wrongly reports a negated mention; the NE gate drops it.
            findings.append({
                "evidence_type": "nepc_diagnosis", "specimen_site": "prostate biopsy",
                "stated_date": None, "confidence": "low",
                "quote": "No small cell component is identified.",
            })
        return json.dumps({"findings": findings}), None


@pytest.fixture
def collected(tmp_path):
    _write_notes(tmp_path)
    collect_nepc_dx_notes.run(_collector_args(tmp_path))
    return tmp_path


def test_collector_keeps_whole_pathology_reports_and_windows_others(collected):
    evidence = pl.read_parquet(collected / "out" / "nepc_evidence.parquet")
    assert set(evidence["DFCI_MRN"]) == {101, 202}
    pathology = evidence.filter(pl.col("note_type") == "Pathology")["snippet"].item()
    assert pathology.startswith("SPECIMEN: Liver, core biopsy")
    assert "INSM1" in pathology
    # Same RPT_ID in two files: note_uid is prefixed by note type.
    assert evidence.filter(pl.col("DFCI_MRN") == 101)["note_uid"].n_unique() == 2
    # The note with no trigger is not evidence.
    assert evidence.height == 3


def test_collectors_require_an_explicit_cohort(tmp_path):
    _write_notes(tmp_path)
    for module in (collect_nepc_dx_notes, collect_avpc_notes):
        with pytest.raises(ValueError, match="prostate-specific"):
            module.run(_collector_args(tmp_path, mrns=None))


def test_end_to_end_labels_grounding_and_gate(collected, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(collected))
    assert len(provider.calls) == 3

    by_mrn = {r["DFCI_MRN"]: r for r in labels.iter_rows(named=True)}
    nepc = by_mrn[101]
    assert nepc["nepc_status"] == "nepc"
    assert nepc["review_status"] == "ok"
    # The clinic note states 2020 (year precision): certainly earlier than the
    # 2021 pathology report, so it is the cited onset.
    assert nepc["nepc_date"] == "2020-01-01"
    assert nepc["nepc_date_precision"] == "year"
    assert nepc["nepc_date_source"] == "stated"
    assert nepc["has_stated_nepc_dx"] is True
    assert set(nepc["evidence_types"]) == {"nepc_diagnosis", "positive_ne_ihc"}

    negated = by_mrn[202]
    assert negated["nepc_status"] == "non_nepc"
    assert negated["has_stated_nepc_dx"] is False
    assert negated["gate_rejections"] == ["nepc_diagnosis:negated_or_hedged:no"]

    assert by_mrn[303]["review_status"] == "no_trigger_notes"
    assert by_mrn[303]["nepc_status"] == "non_nepc"

    out = collected / "out"
    rejected = pl.read_parquet(out / "nepc_rejected_findings.parquet")
    assert sorted(rejected["reason"]) == ["quote_not_in_note", "stated_date_after_note"]
    findings = pl.read_parquet(out / "nepc_findings.parquet")
    pathology = findings.filter(pl.col("note_type") == "Pathology")
    assert pathology.height == 2
    assert set(pathology["date_source"]) == {"note_date"}
    ledger = pl.read_parquet(out / "nepc_processed_notes.parquet")
    assert "ok_with_rejections" in set(ledger["status"])


def test_resume_retry_failures_and_rebuild(collected, monkeypatch):
    failing = FakeProvider(fail_texts={"s/p carboplatin"})
    monkeypatch.setattr(runner, "get_provider", lambda name: failing)
    labels = runner.run(_runner_args(collected))
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["review_status"] == "partial:1/2"
    assert row["nepc_status"] == "nepc"
    assert row["nepc_date"] == "2021-04-02"

    # A plain re-run skips the failed note; nothing is called.
    again = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: again)
    runner.run(_runner_args(collected))
    assert again.calls == []

    runner.run(_runner_args(collected, retry_failures=True))
    assert len(again.calls) == 1
    labels = pl.read_parquet(collected / "out" / "nepc_labels.parquet")
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["review_status"] == "ok"
    assert row["nepc_date"] == "2020-01-01"

    # Rebuild never asks for a provider.
    def no_provider(name):
        raise AssertionError("rebuild must not call the provider")

    monkeypatch.setattr(runner, "get_provider", no_provider)
    rebuilt = runner.run(_runner_args(collected, rebuild_labels_only=True))
    assert rebuilt.equals(labels)
    findings = pl.read_parquet(collected / "out" / "nepc_findings.parquet")
    assert findings.group_by(["note_uid", "finding_index"]).len()["len"].max() == 1


def test_changed_model_requires_overwrite(collected, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    runner.run(_runner_args(collected))
    with pytest.raises(ValueError, match="--overwrite"):
        runner.run(_runner_args(collected, model="model-b"))
    runner.run(_runner_args(collected, model="model-b", overwrite=True))
    assert len(provider.calls) == 6


def test_pilot_mrns_limit_calls_but_label_whole_cohort(collected, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "get_provider", lambda name: provider)
    labels = runner.run(_runner_args(collected, mrns="202"))
    assert {c["note_date"] for c in provider.calls} == {"2020-02-02"}
    status = dict(labels.select("DFCI_MRN", "review_status").iter_rows())
    assert status == {101: "not_run", 202: "ok", 303: "no_trigger_notes"}
    row = labels.filter(pl.col("DFCI_MRN") == 101).row(0, named=True)
    assert row["nepc_status"] is None


def test_validate_finding_rejects_unknown_evidence_type():
    fields, reason = runner.validate_finding({"evidence_type": "vibes"}, {})
    assert fields is None and reason == "invalid_evidence_type:vibes"


def test_unparseable_reply_gets_one_correction_retry():
    class Flaky(FakeProvider):
        def call_with_retry(self, client, model, messages, max_retries):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return "not json", None
            return json.dumps({"findings": []}), None

    provider = Flaky()
    note = {"DFCI_MRN": 1, "note_uid": "Clinician:1", "note_date": "2021-01-01",
            "note_type": "Clinician", "snippet": "small cell"}
    rows, rejected, ledger = note_extraction.extract_note(
        runner.SPEC, provider, None, "m", 1, note
    )
    assert ledger["status"] == "ok" and len(provider.calls) == 2
    assert "failed machine validation" in provider.calls[1][-1]["content"]
