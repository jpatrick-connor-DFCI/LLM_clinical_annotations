"""Output-limit recovery must preserve evidence and leave failures resumable."""

import json
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.config import GLEASON_EVIDENCE_SCHEMA_VERSION, MET_DX_EVIDENCE_SCHEMA_VERSION
from preprocessing.longitudinal import file_sha256, write_scan_config_meta
from tasks.gleason_score import build_gleason_timeline as gleason
from tasks.met_diagnosis import build_met_dx_labels as met


TRUNCATED = "truncated_response: finish_reason=MAX_TOKENS cause=length attempts=3"
NOTE = {
    "note_date": "2021-06-03",
    "note_type": "Clinician",
    "snippet": "Biopsy: Gleason 4+3=7. Metastatic prostate cancer to bone.",
}
GLEASON_FINDING = {
    "primary": 4, "secondary": 3, "total": 7, "grade_group": 3,
    "specimen_type": "biopsy", "scoring_date": None,
    "source_note_date": NOTE["note_date"], "is_historical_reference": False,
    "quote": "Biopsy: Gleason 4+3=7.",
}
CANDIDATE = {
    "evidence_type": "stated_metastatic_disease", "assertion_type": "established",
    "met_site": "bone", "stated_metastasis_date": None,
    "source_note_date": NOTE["note_date"], "modality": "clinical",
    "quote": "Metastatic prostate cancer to bone.", "confidence": "high",
}
SYNTHESIS = {
    **{k: v for k, v in CANDIDATE.items() if k not in {"quote", "stated_metastasis_date"}},
    "has_metastatic_disease": True, "metastasis_date": None,
    "supporting_quote": CANDIDATE["quote"], "rationale": "Established bone metastasis.",
}
ANSWERS = {
    "gleason": {"gleason_findings": [GLEASON_FINDING]},
    "met_map": {"candidates": [CANDIDATE]},
    "met_synthesis": SYNTHESIS,
}


class SequenceProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call_with_retry(self, client, model, messages, max_retries):
        self.calls.append(messages)
        return next(self.responses)


def extract(stage, provider):
    if stage == "gleason":
        return gleason._extract_chunk(provider, None, "model", 1, 123, [NOTE])
    if stage == "met_map":
        return met._extract_chunk(provider, None, "model", 1, 123, 4, [NOTE])
    return met.synthesize_patient(
        provider, None, "model", 1, 123,
        {4: {"candidates": [CANDIDATE]}}, {4: [NOTE]},
    )


@pytest.mark.parametrize("stage", ANSWERS)
@pytest.mark.parametrize("first_response", [
    (None, TRUNCATED),
    ('{"incomplete": [', None),
    ("[]", None),
    ("{}", None),
])
def test_output_correction_recovers_a_grounded_result(stage, first_response):
    provider = SequenceProvider([first_response, (json.dumps(ANSWERS[stage]), None)])
    result, error = extract(stage, provider)

    assert error is None
    if stage == "gleason":
        assert result == [GLEASON_FINDING]
    elif stage == "met_map":
        assert result["candidates"] == [CANDIDATE]
        assert not result["rejected"]
    else:
        assert result["has_metastatic_disease"] is True
        assert result["finding"] == CANDIDATE
    assert len(provider.calls) == 2
    # Same evidence and original instructions; only the correction is appended.
    assert len(provider.calls[0]) == 2
    assert provider.calls[1][:-1] == provider.calls[0]
    correction = provider.calls[1][-1]["content"]
    assert "failed machine validation" in correction
    expected_key = {
        "gleason": "gleason_findings", "met_map": "candidates",
        "met_synthesis": "has_metastatic_disease",
    }[stage]
    assert expected_key in correction


@pytest.mark.parametrize("stage", ANSWERS)
@pytest.mark.parametrize("failure", [
    (None, TRUNCATED),
    ('{"incomplete": [', None),
])
def test_persistent_output_failure_stops_after_one_correction(stage, failure):
    provider = SequenceProvider([failure, failure])
    result, error = extract(stage, provider)
    assert result is None
    assert error.startswith("truncated_response" if failure[1] else "json_parse")
    assert len(provider.calls) == 2


@pytest.mark.parametrize("stage", ANSWERS)
@pytest.mark.parametrize("error", ["resource_exhausted: 429", "content_filter_response"])
def test_non_output_errors_do_not_trigger_correction(stage, error):
    provider = SequenceProvider([(None, error)])
    assert extract(stage, provider) == (None, error)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("stage", ANSWERS)
def test_successful_output_needs_no_correction(stage):
    provider = SequenceProvider([(json.dumps(ANSWERS[stage]), None)])
    assert extract(stage, provider)[1] is None
    assert len(provider.calls) == 1


@pytest.mark.parametrize("task,prefix,schema_version", [
    (gleason, "gleason", GLEASON_EVIDENCE_SCHEMA_VERSION),
    (met, "met_dx", MET_DX_EVIDENCE_SCHEMA_VERSION),
])
def test_failed_chunks_resume_without_recalling_successful_chunks(
    task, prefix, schema_version, tmp_path, monkeypatch,
):
    evidence_path = tmp_path / f"{prefix}_evidence.parquet"
    dates = ["2021-06-03", "2021-07-04"]
    pl.DataFrame([
        {"DFCI_MRN": 123, "chunk_index": index, **NOTE, "note_date": date}
        for index, date in zip([2, 7], dates)
    ]).write_parquet(evidence_path)
    write_scan_config_meta(
        tmp_path / f"{prefix}_evidence.meta.parquet", "scan-1",
        evidence_sha256=file_sha256(evidence_path),
        evidence_schema_version=schema_version, cohort_mrns=[123],
    )

    class Provider:
        default_model = "model"
        failing = True

        def __init__(self):
            self.calls = []

        def build_client(self):
            return None

        def call_with_retry(self, client, model, messages, max_retries):
            payload = json.loads(messages[1]["content"])
            if "candidates" in payload:
                self.calls.append("synthesis")
                return json.dumps(SYNTHESIS), None
            date = payload["notes"][0]["note_date"]
            self.calls.append(date)
            if self.failing and date == dates[1]:
                return None, TRUNCATED
            key, finding = (
                ("gleason_findings", GLEASON_FINDING) if task is gleason
                else ("candidates", CANDIDATE)
            )
            return json.dumps({key: [{**finding, "source_note_date": date}]}), None

    provider = Provider()
    monkeypatch.setattr(task, "get_provider", lambda name: provider)
    args = SimpleNamespace(
        output_dir=tmp_path, evidence_path=evidence_path, evidence_meta_path=None,
        mrn_file=None, mrns=None, provider="vertex_ai", model=None, max_workers=1,
        max_retries=1, limit_patients=None, rebuild_labels_only=False, overwrite=False,
    )
    task.run(args)
    log_path = tmp_path / f"{prefix}_processed_chunks.parquet"
    log = pl.read_parquet(log_path)
    failed = log.filter(pl.col("chunk_index") == 7).to_dicts()[0]
    assert failed["status"].startswith("truncated_response")
    assert (123, 7) not in task.read_done_chunks(log_path)
    assert provider.calls == [dates[0], dates[1], dates[1]]

    provider.failing = False
    task.run(args)
    assert provider.calls.count(dates[0]) == 1
    assert provider.calls.count(dates[1]) == 3
    assert task.read_done_chunks(log_path) == {(123, 2), (123, 7)}
    if task is gleason:
        assert pl.read_parquet(tmp_path / "gleason_timeline.parquet").height == 2
    else:
        labels = pl.read_parquet(tmp_path / "met_dx_labels.parquet")
        assert labels["has_metastatic_disease"].to_list() == [True]
    calls = list(provider.calls)
    task.run(args)
    assert provider.calls == calls
