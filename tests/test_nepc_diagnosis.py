"""Tests for the strict-precision NEPC diagnosis-date task.

The veto and trigger tests below are the executable form of the precision
contract: each case corresponds to a false-positive class observed in the
recall-biased longitudinal pipeline.
"""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from preprocessing.config import NEPC_DX_EVIDENCE_SCHEMA_VERSION
from preprocessing.longitudinal import file_sha256, write_scan_config_meta
from preprocessing.triggers import find_trigger_matches
from tasks.nepc_diagnosis import build_nepc_dx_labels as dx
from tasks.nepc_diagnosis.prompts import (
    NEPC_DX_MAP_PROMPT,
    NEPC_DX_SYNTHESIS_PROMPT,
)
from tasks.nepc_diagnosis.veto import screen_quote

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_collector():
    """Import the collector by path -- preprocessing/cli is not a package."""
    path = REPO_ROOT / "preprocessing" / "cli" / "collect_nepc_dx_notes.py"
    spec = importlib.util.spec_from_file_location("collect_nepc_dx_notes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DX_TRIGGERS = _load_collector().TRIGGER_REGEX


# --------------------------------------------------------------------------
# Trigger retrieval
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Final diagnosis: small cell carcinoma of the prostate.",
        "Biopsy shows transformation to neuroendocrine carcinoma.",
        "Patient has t-NEPC.",
        "Oat cell carcinoma involving the prostate.",
        "Histologic transformation was documented.",
        "Tumor cells are positive for synaptophysin.",
    ],
)
def test_narrow_triggers_match_diagnostic_language(text):
    assert find_trigger_matches(text, DX_TRIGGERS)


@pytest.mark.parametrize(
    "text",
    [
        # AVPC atomic families, dropped wholesale.
        "PSA 7.2 ng/mL",
        "There is a 6.3 cm pelvic mass.",
        "Hepatic metastases are present.",
        "Castration-resistant prostate cancer.",
        "Gleason 4+5 adenocarcinoma.",
        # Bare transformation wording.
        "Discussed transformation of the care plan with the patient.",
        # Standalone NSE, dedifferentiation, lineage plasticity.
        "NSE 12",
        "Dedifferentiated liposarcoma of the thigh.",
        "Lineage plasticity is an area of active research.",
    ],
)
def test_narrow_triggers_drop_avpc_and_noise_language(text):
    assert not find_trigger_matches(text, DX_TRIGGERS)


# --------------------------------------------------------------------------
# Deterministic quote gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "quote",
    [
        "Final diagnosis: small cell carcinoma of the prostate.",
        "Biopsy showed transformation to neuroendocrine carcinoma.",
        # Guards against over-vetoing: "features" must not be a hedge cue.
        "Small cell carcinoma of the prostate with neuroendocrine features.",
        # A clean assertion followed by an unrelated negation must survive.
        "Final diagnosis: small cell carcinoma; no evidence of nodal disease.",
    ],
)
def test_veto_accepts_stated_diagnoses(quote):
    ok, reason = screen_quote(quote)
    assert ok, reason


@pytest.mark.parametrize(
    "quote",
    [
        "There is no small cell carcinoma.",
        "Negative for neuroendocrine carcinoma.",
        "Findings are suspicious for small cell carcinoma.",
        "The differential includes neuroendocrine carcinoma.",
        "Rule out small cell carcinoma.",
        "Cannot exclude small cell carcinoma.",
        # Non-prostate primary -- the largest confounder for this trigger set.
        "Diagnosis: small cell carcinoma of the lung.",
    ],
)
def test_veto_rejects_negated_and_hedged_quotes(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("negated_or_hedged:")


@pytest.mark.parametrize(
    "quote",
    [
        "Focal neuroendocrine differentiation is seen.",
        "Neuroendocrine features are present.",
    ],
)
def test_veto_rejects_feature_only_quotes(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "no_diagnostic_assertion"


@pytest.mark.parametrize(
    "quote",
    [
        "Prostatic adenocarcinoma, Gleason 4+5.",
        # IHC markers are retrieval triggers only -- they are not disease terms,
        # so a stain-only quote carries nothing to adjudicate.
        "Tumor cells are positive for synaptophysin and chromogranin.",
        "INSM1 positive; CD56 reactive.",
        "",
        None,
    ],
)
def test_veto_rejects_quotes_without_a_disease_term(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "no_nepc_term"


@pytest.mark.parametrize(
    "quote",
    [
        "Inclusion criteria: patients with small cell carcinoma of the prostate are eligible.",
        "Exclusion criteria: history of small cell neuroendocrine carcinoma.",
        "This is a Phase II study of patients with neuroendocrine prostate cancer.",
        "Eligible subjects must have histologically confirmed neuroendocrine carcinoma.",
        "Patient consented to DFCI 19-002 for small cell prostate cancer.",
        "Protocol: A study of lurbinectedin in small cell carcinoma.",
        "Cohort B: neuroendocrine carcinoma of the prostate.",
        "Enrolling patients with treatment-emergent neuroendocrine prostate cancer.",
    ],
)
def test_veto_rejects_clinical_trial_boilerplate(quote):
    """Stock trial text describes an eligible population, not this patient.

    It is a systematic false-positive source: an NEPC trial's eligibility block
    selects for exactly this wording, and it is copy-forwarded across notes.
    """
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("boilerplate:")


@pytest.mark.parametrize(
    "quote",
    [
        # Guards against over-vetoing: a real diagnosis is often the REASON a
        # trial is discussed, and both land in the same note.
        "FINAL DIAGNOSIS: Small cell carcinoma of the prostate.",
        "Pathologic diagnosis of small cell carcinoma of the prostate confirmed"
        " on the biopsy specimen.",
    ],
)
def test_veto_keeps_genuine_diagnoses_near_trial_context(quote):
    ok, reason = screen_quote(quote)
    assert ok, reason


@pytest.mark.parametrize(
    "quote",
    [
        # Clinician assertion contexts: for many patients the diagnosis is
        # recorded only in an oncology progress note (outside pathology, or a
        # transformation established across visits), so these must qualify.
        "ASSESSMENT: Metastatic neuroendocrine prostate cancer.",
        "IMPRESSION: Small cell carcinoma of the prostate.",
        "Problem list: neuroendocrine prostate carcinoma.",
        "He has a known diagnosis of small cell prostate cancer.",
        "68 yo man s/p chemo for small cell prostate carcinoma.",
        "Dx: neuroendocrine prostate cancer.",
        # Self-anchoring acronyms: "NEPC" already denotes a carcinoma, so no
        # separate "carcinoma"/"cancer" word is required.
        "Patient with t-NEPC on carboplatin/etoposide.",
        "Assessment: NEPC, on treatment.",
        "Metastatic NEPC.",
        "Known SCPC with liver mets.",
    ],
)
def test_veto_accepts_clinician_stated_diagnoses(quote):
    ok, reason = screen_quote(quote)
    assert ok, reason


@pytest.mark.parametrize(
    "quote",
    [
        # Surveillance/anticipatory wording is a clinical-note idiom, and the
        # hedge cues must apply to progress notes exactly as to pathology.
        "We will monitor for NEPC.",
        "Surveillance for transformation to NEPC.",
        "Watching for NEPC.",
        "If he develops NEPC we would consider platinum.",
        "Will assess for neuroendocrine transformation.",
        "Concern for progression to NEPC.",
        "Discussed the risk of transformation to NEPC.",
    ],
)
def test_veto_rejects_clinical_surveillance_wording(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("negated_or_hedged:")


def test_veto_rejects_postposed_negation():
    ok, reason = screen_quote(
        "Small cell carcinoma is not present in this specimen."
    )
    assert not ok
    assert reason == "negated_or_hedged:post_negation"


# --------------------------------------------------------------------------
# Prompt contract
# --------------------------------------------------------------------------

def test_prompts_state_the_strict_exclusions():
    combined = f"{NEPC_DX_MAP_PROMPT}\n{NEPC_DX_SYNTHESIS_PROMPT}"
    assert "Immunohistochemistry results alone" in combined
    assert '"Neuroendocrine features"' in combined
    assert "When in doubt, report nothing." in combined
    assert "eligibility criteria" in combined
    assert "Pathology reports and clinician progress notes are BOTH" in combined
    assert "NEVER copy the note date into this field" in combined
    assert "Never introduce a quote, a date, or a fact" in combined


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _chunk():
    return [
        {
            "note_date": "2021-06-03",
            "note_type": "Pathology",
            "snippet": (
                "FINAL DIAGNOSIS: Small cell carcinoma of the prostate. "
                "Tumor cells are positive for synaptophysin and chromogranin."
            ),
        }
    ]


def _candidate(**overrides):
    item = {
        "evidence_type": "stated_diagnosis",
        "assertion_type": "established",
        "stated_diagnosis_date": None,
        "source_note_date": "2021-06-03",
        "modality": "pathology",
        "quote": "FINAL DIAGNOSIS: Small cell carcinoma of the prostate.",
        "confidence": "high",
    }
    item.update(overrides)
    return item


def test_valid_candidate_is_kept():
    result, fatal = dx.validate_map_result({"candidates": [_candidate()]}, _chunk())
    assert fatal is None
    assert len(result["candidates"]) == 1
    assert result["rejected"] == {}


def test_invented_quote_is_rejected():
    result, fatal = dx.validate_map_result(
        {"candidates": [_candidate(quote="Diagnosis: neuroendocrine carcinoma of the prostate.")]},
        _chunk(),
    )
    assert fatal is None
    assert result["candidates"] == []
    assert "quote_or_source_date_not_in_evidence" in result["rejected"]


def test_quote_grounded_in_a_different_note_date_is_rejected_not_redated():
    """The key divergence from the recall-biased longitudinal pipeline.

    build_nepc_timeline._find_support_note would silently re-date this finding to
    2021-06-03. Here a provenance mismatch is a rejection.
    """
    result, fatal = dx.validate_map_result(
        {"candidates": [_candidate(source_note_date="2019-01-01")]}, _chunk()
    )
    assert fatal is None
    assert result["candidates"] == []
    assert "quote_or_source_date_not_in_evidence" in result["rejected"]


def test_ihc_only_candidate_rejected_while_diagnostic_candidate_survives():
    """Partial success: one bad candidate must not discard the chunk's good one."""
    result, fatal = dx.validate_map_result(
        {
            "candidates": [
                _candidate(
                    quote="Tumor cells are positive for synaptophysin and chromogranin."
                ),
                _candidate(),
            ]
        },
        _chunk(),
    )
    assert fatal is None
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["quote"].startswith("FINAL DIAGNOSIS")
    assert result["rejected"] == {"no_nepc_term": 1}


def test_diagnosis_date_after_source_note_is_rejected():
    result, _ = dx.validate_map_result(
        {"candidates": [_candidate(stated_diagnosis_date="2023-01-01")]}, _chunk()
    )
    assert "diagnosis_date_after_source_note" in result["rejected"]


@pytest.mark.parametrize(
    "field,value,prefix",
    [
        ("evidence_type", "ihc", "invalid_evidence_type"),
        ("assertion_type", "maybe", "invalid_assertion_type"),
        ("modality", "labs", "invalid_modality"),
        ("confidence", "certain", "invalid_confidence"),
    ],
)
def test_invalid_enums_are_rejected(field, value, prefix):
    result, _ = dx.validate_map_result(
        {"candidates": [_candidate(**{field: value})]}, _chunk()
    )
    assert any(reason.startswith(prefix) for reason in result["rejected"])


def test_structural_problems_are_fatal_to_the_chunk():
    assert dx.validate_map_result({"nope": []}, _chunk())[1] == "missing_candidates"
    assert dx.validate_map_result([], _chunk())[1].startswith("non_dict_response")


def test_duplicate_candidates_are_collapsed():
    result, _ = dx.validate_map_result(
        {"candidates": [_candidate(), _candidate()]}, _chunk()
    )
    assert len(result["candidates"]) == 1


def _synth(**overrides):
    payload = {
        "has_nepc_diagnosis": True,
        "evidence_type": "stated_diagnosis",
        "assertion_type": "established",
        "diagnosis_date": None,
        "source_note_date": "2021-06-03",
        "modality": "pathology",
        "supporting_quote": "FINAL DIAGNOSIS: Small cell carcinoma of the prostate.",
        "confidence": "high",
        "rationale": "Pathology states the diagnosis.",
    }
    payload.update(overrides)
    return payload


def _index():
    return {("final diagnosis: small cell carcinoma of the prostate.", "2021-06-03")}


def test_synthesis_positive_is_accepted():
    result, fatal = dx.validate_synthesis_result(_synth(), _chunk(), _index())
    assert fatal is None
    assert result["has_nepc_diagnosis"] is True
    assert result["finding"]["source_note_date"] == "2021-06-03"


def test_synthesis_quote_not_from_a_candidate_downgrades_to_negative():
    """Adjudication may not invent evidence; it is downgraded, never silently kept."""
    result, fatal = dx.validate_synthesis_result(_synth(), _chunk(), set())
    assert fatal is None
    assert result["has_nepc_diagnosis"] is False
    assert result["finding"] is None
    assert result["rejected"] == {"quote_not_from_candidate": 1}


def test_synthesis_non_bool_label_is_fatal():
    result, fatal = dx.validate_synthesis_result(
        _synth(has_nepc_diagnosis="yes"), _chunk(), _index()
    )
    assert result is None
    assert fatal.startswith("invalid_has_nepc_diagnosis")


def test_synthesis_negative_keeps_its_verdict():
    result, fatal = dx.validate_synthesis_result(
        {
            "has_nepc_diagnosis": False,
            "evidence_type": None,
            "assertion_type": None,
            "diagnosis_date": None,
            "source_note_date": None,
            "modality": None,
            "supporting_quote": None,
            "rationale": "Every mention was an IHC result.",
        },
        _chunk(),
        _index(),
    )
    assert fatal is None
    assert result["has_nepc_diagnosis"] is False
    assert result["rationale"] == "Every mention was an IHC result."


def test_negative_carrying_evidence_fields_is_flagged_but_still_negative():
    result, _ = dx.validate_synthesis_result(
        _synth(has_nepc_diagnosis=False), _chunk(), _index()
    )
    assert result["has_nepc_diagnosis"] is False
    assert result["rejected"] == {"negative_with_evidence_fields": 1}


# --------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------

def test_fingerprint_changes_with_model():
    a = dx.extraction_run_config("scan-1", "dfci_gpt", "model-a")
    b = dx.extraction_run_config("scan-1", "dfci_gpt", "model-b")
    assert a != b


def test_fingerprint_changes_with_veto_version(monkeypatch):
    """Changing the deterministic gate must force --overwrite."""
    before = dx.extraction_run_config("scan-1", "dfci_gpt", "model-a")
    monkeypatch.setattr(dx, "VETO_VERSION", "veto-sentinel-not-a-real-version")
    assert dx.extraction_run_config("scan-1", "dfci_gpt", "model-a") != before


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def _write_evidence(tmp_path, snippet, note_date="2021-06-03", mrn=123):
    evidence = tmp_path / "nepc_dx_evidence.parquet"
    pl.DataFrame(
        {
            "DFCI_MRN": [mrn],
            "chunk_index": [0],
            "note_date": [note_date],
            "note_type": ["Pathology"],
            "snippet": [snippet],
        }
    ).write_parquet(evidence)
    write_scan_config_meta(
        tmp_path / "nepc_dx_evidence.meta.parquet",
        "scan-1",
        evidence_sha256=file_sha256(evidence),
        evidence_schema_version=NEPC_DX_EVIDENCE_SCHEMA_VERSION,
        cohort_mrns=[mrn, 999],
    )
    return evidence


def _args(tmp_path, evidence, **overrides):
    base = dict(
        output_dir=tmp_path,
        evidence_path=evidence,
        evidence_meta_path=None,
        mrn_file=None,
        mrns=None,
        provider="vertex_ai",
        model=None,
        max_workers=2,
        max_retries=1,
        limit_patients=None,
        rebuild_labels_only=False,
        overwrite=False,
    )
    base.update(overrides)
    return Namespace(**base)


class _Provider:
    """Fake provider: maps the chunk, then adjudicates from the candidates."""

    default_model = "model-a"

    def __init__(self, stated_date=None):
        self.calls = 0
        self.stated_date = stated_date

    def build_client(self):
        return object()

    def call_with_retry(self, client, model, messages, max_retries):
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        if "candidates" in payload:
            candidate = payload["candidates"][0]
            return json.dumps(
                {
                    "has_nepc_diagnosis": True,
                    "evidence_type": candidate["evidence_type"],
                    "assertion_type": candidate["assertion_type"],
                    "diagnosis_date": self.stated_date,
                    "source_note_date": candidate["source_note_date"],
                    "modality": candidate["modality"],
                    "supporting_quote": candidate["quote"],
                    "confidence": "high",
                    "rationale": "Pathology states the diagnosis.",
                }
            ), None
        note = payload["notes"][0]
        return json.dumps(
            {
                "candidates": [
                    {
                        "evidence_type": "stated_diagnosis",
                        "assertion_type": "established",
                        "stated_diagnosis_date": self.stated_date,
                        "source_note_date": note["note_date"],
                        "modality": "pathology",
                        "quote": note["note_text"],
                        "confidence": "high",
                    }
                ]
            }
        ), None


DX_SNIPPET = "FINAL DIAGNOSIS: Small cell carcinoma of the prostate."


def test_undated_positive_falls_back_to_the_note_date_and_is_kept(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    provider = _Provider()
    monkeypatch.setattr(dx, "get_provider", lambda name: provider)
    dx.run(_args(tmp_path, evidence))

    labels = pl.read_parquet(tmp_path / "nepc_dx_labels.parquet")
    row = labels.filter(pl.col("DFCI_MRN") == 123).to_dicts()[0]
    assert row["has_nepc_diagnosis"] is True
    assert row["date_source"] == "note_date"
    assert row["diagnosis_date"] == "2021-06-03"
    assert row["label_source"] == "adjudicated"
    assert row["supporting_quote"] == DX_SNIPPET


def test_stated_date_wins_and_is_marked_stated(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    provider = _Provider(stated_date="2021-03-01")
    monkeypatch.setattr(dx, "get_provider", lambda name: provider)
    dx.run(_args(tmp_path, evidence))

    row = (
        pl.read_parquet(tmp_path / "nepc_dx_labels.parquet")
        .filter(pl.col("DFCI_MRN") == 123)
        .to_dicts()[0]
    )
    assert row["date_source"] == "stated"
    assert row["diagnosis_date"] == "2021-03-01"


def test_cohort_patients_without_evidence_become_auto_negatives(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    monkeypatch.setattr(dx, "get_provider", lambda name: _Provider())
    dx.run(_args(tmp_path, evidence))

    row = (
        pl.read_parquet(tmp_path / "nepc_dx_labels.parquet")
        .filter(pl.col("DFCI_MRN") == 999)
        .to_dicts()[0]
    )
    assert row["has_nepc_diagnosis"] is False
    assert row["label_source"] == "auto_negative_no_evidence"
    assert row["diagnosis_date"] is None


def test_vetoed_evidence_yields_no_positive_and_an_audit_row(tmp_path, monkeypatch):
    """A negated note must not produce a positive, however the model answers."""
    evidence = _write_evidence(
        tmp_path, "FINAL DIAGNOSIS: No evidence of small cell carcinoma."
    )
    monkeypatch.setattr(dx, "get_provider", lambda name: _Provider())
    dx.run(_args(tmp_path, evidence))

    labels = pl.read_parquet(tmp_path / "nepc_dx_labels.parquet")
    assert labels.filter(pl.col("has_nepc_diagnosis")).height == 0
    rejected = pl.read_parquet(tmp_path / "nepc_dx_rejected_findings.parquet")
    assert any(
        str(reason).startswith("negated_or_hedged:")
        for reason in rejected["reason"].to_list()
    )


def test_resume_and_config_guards(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    provider = _Provider()
    monkeypatch.setattr(dx, "get_provider", lambda name: provider)
    args = _args(tmp_path, evidence)

    dx.run(args)
    assert provider.calls == 2  # one map + one adjudication

    dx.run(args)
    assert provider.calls == 2  # fully resumed, zero new calls

    with pytest.raises(ValueError, match="Provider, model, prompt"):
        dx.run(_args(tmp_path, evidence, model="model-b"))

    monkeypatch.setattr(dx, "VETO_VERSION", "veto-sentinel-not-a-real-version")
    with pytest.raises(ValueError, match="Provider, model, prompt"):
        dx.run(args)


def test_evidence_content_change_is_caught(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    monkeypatch.setattr(dx, "get_provider", lambda name: _Provider())
    dx.run(_args(tmp_path, evidence))

    pl.DataFrame(
        {
            "DFCI_MRN": [123],
            "chunk_index": [0],
            "note_date": ["2021-06-03"],
            "note_type": ["Pathology"],
            "snippet": ["changed evidence"],
        }
    ).write_parquet(evidence)
    with pytest.raises(ValueError, match="Evidence content"):
        dx.run(_args(tmp_path, evidence))


def test_rebuild_labels_only_makes_no_provider_calls(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    provider = _Provider()
    monkeypatch.setattr(dx, "get_provider", lambda name: provider)
    dx.run(_args(tmp_path, evidence))
    before = provider.calls

    (tmp_path / "nepc_dx_labels.parquet").unlink()
    dx.run(_args(tmp_path, evidence, rebuild_labels_only=True))
    assert provider.calls == before
    assert pl.read_parquet(tmp_path / "nepc_dx_labels.parquet").height == 2


def test_rebuild_labels_only_conflicts_with_overwrite(tmp_path, monkeypatch):
    evidence = _write_evidence(tmp_path, DX_SNIPPET)
    monkeypatch.setattr(dx, "get_provider", lambda name: _Provider())
    with pytest.raises(ValueError, match="cannot be combined"):
        dx.run(_args(tmp_path, evidence, rebuild_labels_only=True, overwrite=True))
