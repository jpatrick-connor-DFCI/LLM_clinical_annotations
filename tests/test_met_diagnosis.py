"""Tests for the metastatic prostate cancer label + first-mention date task.

The veto and trigger tests are the executable form of the precision contract.
Two boundaries carry most of the weight here and have dedicated sections:

  - M1 vs. N1: regional pelvic nodal disease alone is NOT metastatic, despite
    pathology reports describing it with the word "metastatic".
  - destination vs. primary: "metastatic prostate cancer to the lung" names a
    lung LESION and qualifies, while "known metastatic lung primary" names a
    lung CANCER and does not. The NEPC gate never had to make this distinction,
    since there a nearby non-prostate site was always a competing primary.
"""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from preprocessing.config import MET_DX_EVIDENCE_SCHEMA_VERSION
from preprocessing.longitudinal import file_sha256, write_scan_config_meta
from preprocessing.triggers import combined_text_pattern, find_trigger_matches
from tasks.met_diagnosis import build_met_dx_labels as md
from tasks.met_diagnosis.prompts import (
    MET_DX_MAP_PROMPT,
    MET_DX_SYNTHESIS_PROMPT,
)
from tasks.met_diagnosis.veto import screen_quote

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_collector():
    """Import the collector by path -- preprocessing/cli is not a package."""
    path = REPO_ROOT / "preprocessing" / "cli" / "collect_met_dx_notes.py"
    spec = importlib.util.spec_from_file_location("collect_met_dx_notes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MET_TRIGGERS = _load_collector().TRIGGER_REGEX


# --------------------------------------------------------------------------
# Trigger retrieval
#
# Retrieval is deliberately broad: it fires on every "no evidence of metastatic
# disease" too. Precision is veto.py's job downstream, not the scan's.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "ASSESSMENT: metastatic castration-resistant prostate cancer.",
        "Widespread bone mets.",
        "Clinical stage T3bN0M1b.",
        "Stage IV prostate cancer.",
        "IMPRESSION: innumerable sclerotic osseous lesions.",
        "Bone scan shows increased uptake.",
        "There is no evidence of distant spread.",
        "Diffuse osseous involvement.",
    ],
)
def test_met_triggers_match_metastasis_language(text):
    assert find_trigger_matches(text, MET_TRIGGERS)


@pytest.mark.parametrize(
    "text",
    [
        "PSA 7.2 ng/mL",
        "Gleason 4+3 adenocarcinoma.",
        "S/p radical prostatectomy in 2018.",
        "Castration-resistant prostate cancer.",
        "Continue leuprolide every 3 months.",
        "Clinical stage T2cN0M0.",
        # Not a TNM string -- the M-category patterns must not fire on it.
        "Room 3M1 on the ward.",
    ],
)
def test_met_triggers_drop_unrelated_prostate_language(text):
    assert not find_trigger_matches(text, MET_TRIGGERS)


@pytest.mark.parametrize(
    "text,expected",
    [
        # A contiguous TNM string has no word boundary anywhere inside it, so a
        # plain \bm1\b never matches one. These are the explicitly-staged
        # metastatic patients, so missing them would be a silent recall hole.
        ("Clinical stage T3bN0M1b.", True),
        ("Stage T2cN1M1.", True),
        ("pT3aN1M1c disease.", True),
        ("ypT2N0M1.", True),
        ("He has M1b prostate cancer.", True),
        ("Clinical stage T2cN0M0.", False),
        ("Room 3M1 on the ward.", False),
    ],
)
def test_trigger_pattern_runs_under_the_polars_regex_engine(text, expected):
    """The collector's pattern must be valid for RUST regex, not just Python re.

    combined_text_pattern() is pushed into scan_parquet's predicate, where
    Polars compiles it with Rust's regex crate. That crate rejects lookaround
    outright, so a pattern using it parses fine under Python's re -- and under
    find_trigger_matches -- while failing at load time against real parquets.
    Exercising the pattern through Polars is the only way this file can catch
    that class of defect.
    """
    pattern = combined_text_pattern(MET_TRIGGERS)
    got = (
        pl.DataFrame({"t": [text]})
        .select(pl.col("t").str.contains(pattern))["t"]
        .to_list()[0]
    )
    assert got is expected


# --------------------------------------------------------------------------
# Deterministic quote gate -- accepts
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "quote",
    [
        "IMPRESSION: multiple osseous metastases consistent with metastatic prostate cancer.",
        "ASSESSMENT: mCRPC with liver metastases.",
        "Problem list: metastatic prostate cancer to bone.",
        "Bone scan demonstrates widespread osseous metastases.",
        "He has M1b prostate cancer.",
        "s/p radium-223 for osseous metastases.",
        "IMPRESSION: innumerable sclerotic osseous lesions consistent with"
        " metastatic disease.",
        "Pathology: metastatic adenocarcinoma in retroperitoneal lymph node,"
        " consistent with prostate primary.",
    ],
)
def test_veto_accepts_asserted_metastatic_disease(quote):
    ok, reason = screen_quote(quote)
    assert ok, reason


def test_veto_keeps_stable_metastatic_disease():
    """Copy-forward guard.

    "stable"/"unchanged" are status words in an oncology assessment, not
    negations -- "stable metastatic disease on abiraterone" is one of the most
    common ways a positive is written, and vetoing it would lose a large share
    of the cohort.
    """
    ok, reason = screen_quote("Stable metastatic disease on abiraterone.")
    assert ok, reason


def test_veto_keeps_metastasis_to_a_non_prostate_site():
    """A named organ near the term is usually a DESTINATION, not a primary."""
    ok, reason = screen_quote("Metastatic prostate cancer to the lung.")
    assert ok, reason


# --------------------------------------------------------------------------
# Deterministic quote gate -- rejects
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "quote",
    [
        "No evidence of metastatic disease.",
        "Bone scan negative for osseous metastases.",
        "Lesion suspicious for osseous metastasis.",
        "Findings concerning for metastatic disease.",
        "Cannot exclude metastatic disease.",
        "Mild degenerative changes; no evidence of osseous metastatic disease.",
        "Monitoring for metastatic progression.",
        "At risk for metastatic disease.",
        "Restaging to rule out metastasis.",
    ],
)
def test_veto_rejects_negated_and_hedged_quotes(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("negated_or_hedged:")


@pytest.mark.parametrize(
    "quote",
    [
        # A site descriptor with nothing asserting spread.
        "Sclerotic lesions in the pelvis.",
        "No new osseous lesions.",
    ],
)
def test_veto_rejects_site_descriptors_without_an_assertion(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "no_metastatic_assertion"


def test_veto_rejects_uptake_without_a_site_term():
    """"Increased uptake in the right rib" carries no met term at all.

    It is retrieved by the "bone scan" trigger, not by a metastasis term, so it
    fails at the first gate rather than the assertion gate.
    """
    ok, reason = screen_quote("Increased uptake in the right rib.")
    assert not ok
    assert reason == "no_met_term"


@pytest.mark.parametrize(
    "quote",
    [
        "Sclerotic focus likely degenerative.",
        "PSA 12.4; continue ADT.",
        "",
        None,
    ],
)
def test_veto_rejects_quotes_without_a_met_term(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "no_met_term"


@pytest.mark.parametrize(
    "quote",
    [
        # N1, not M1 -- the sharpest boundary in this task. Prostatectomy
        # pathology describes regional nodal involvement with the word
        # "metastatic", and treating it as M1 would mislabel a large number of
        # locally advanced, non-metastatic patients.
        "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes.",
        "Metastatic carcinoma involving obturator lymph nodes.",
        "Metastatic disease in the internal iliac nodes.",
    ],
)
def test_veto_rejects_regional_nodal_only_disease(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "regional_nodal_only"


def test_regional_nodes_alongside_a_distant_site_still_qualify():
    """The N1 check is a property of the quote's whole site set."""
    ok, reason = screen_quote(
        "Metastatic disease involving pelvic and retroperitoneal lymph nodes."
    )
    assert ok, reason


def test_regional_nodes_with_bone_disease_still_qualify():
    ok, reason = screen_quote(
        "Metastatic prostate cancer involving pelvic nodes and the lumbar spine."
    )
    assert ok, reason


@pytest.mark.parametrize(
    "quote",
    [
        "Known metastatic lung primary with osseous lesions.",
        "History of metastatic colon cancer.",
        "Metastatic melanoma, primary resected in 2015.",
    ],
)
def test_veto_rejects_a_non_prostate_primary(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("non_prostate_primary:")


@pytest.mark.parametrize(
    "quote",
    [
        "Inclusion criteria: patients with metastatic castration-resistant"
        " prostate cancer are eligible.",
        "This is a Phase III study of patients with metastatic prostate cancer.",
        "Cohort B: metastatic hormone-sensitive prostate cancer.",
    ],
)
def test_veto_rejects_clinical_trial_boilerplate(quote):
    """Trial text names metastatic disease because the trial targets it."""
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("boilerplate:")


# --------------------------------------------------------------------------
# Prompt contract
# --------------------------------------------------------------------------

def test_prompts_state_the_strict_exclusions():
    combined = f"{MET_DX_MAP_PROMPT}\n{MET_DX_SYNTHESIS_PROMPT}"
    assert "REGIONAL PELVIC NODAL DISEASE ALONE" in combined
    assert "N1 disease, NOT M1" in combined
    assert "When in doubt, report nothing." in combined
    assert "inclusion criteria" in combined.lower()
    assert "NEVER copy the note date into this field" in combined
    assert "Never introduce a quote, a date, or a fact" in combined
    assert "none outranks the others" in combined


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

MET_SNIPPET = "IMPRESSION: multiple osseous metastases consistent with metastatic prostate cancer."


def _chunk():
    return [
        {
            "note_date": "2021-06-03",
            "note_type": "Imaging",
            "snippet": MET_SNIPPET + " No new pulmonary nodules.",
        }
    ]


def _candidate(**overrides):
    item = {
        "evidence_type": "imaging_metastasis",
        "assertion_type": "established",
        "met_site": "bone",
        "stated_metastasis_date": None,
        "source_note_date": "2021-06-03",
        "modality": "imaging",
        "quote": MET_SNIPPET,
        "confidence": "high",
    }
    item.update(overrides)
    return item


def test_valid_candidate_is_kept():
    result, fatal = md.validate_map_result({"candidates": [_candidate()]}, _chunk())
    assert fatal is None
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["met_site"] == "bone"
    assert result["rejected"] == {}


def test_invented_quote_is_rejected():
    result, fatal = md.validate_map_result(
        {"candidates": [_candidate(quote="Diffuse hepatic metastases are present.")]},
        _chunk(),
    )
    assert fatal is None
    assert result["candidates"] == []
    assert "quote_or_source_date_not_in_evidence" in result["rejected"]


def test_quote_grounded_in_a_different_note_date_is_rejected_not_redated():
    """A provenance mismatch is a rejection, never a silent re-dating.

    The date IS the deliverable here, so correcting a finding's note date would
    corrupt the answer rather than repair it.
    """
    result, fatal = md.validate_map_result(
        {"candidates": [_candidate(source_note_date="2019-01-01")]}, _chunk()
    )
    assert fatal is None
    assert result["candidates"] == []
    assert "quote_or_source_date_not_in_evidence" in result["rejected"]


def test_met_site_must_be_visible_in_the_grounded_quote():
    """A bone-only quote cannot carry a visceral claim."""
    result, _ = md.validate_map_result(
        {"candidates": [_candidate(met_site="visceral")]}, _chunk()
    )
    assert result["candidates"] == []
    assert result["rejected"] == {"met_site_not_in_quote": 1}


def test_unspecified_met_site_is_exempt_from_grounding():
    result, _ = md.validate_map_result(
        {"candidates": [_candidate(met_site="unspecified")]}, _chunk()
    )
    assert len(result["candidates"]) == 1


def test_vetoed_candidate_rejected_while_valid_candidate_survives():
    """Partial success: one bad candidate must not discard the chunk's good one."""
    result, fatal = md.validate_map_result(
        {
            "candidates": [
                _candidate(quote="No new pulmonary nodules.", met_site="visceral"),
                _candidate(),
            ]
        },
        _chunk(),
    )
    assert fatal is None
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["quote"] == MET_SNIPPET


def test_metastasis_date_after_source_note_is_rejected():
    result, _ = md.validate_map_result(
        {"candidates": [_candidate(stated_metastasis_date="2023-01-01")]}, _chunk()
    )
    assert "metastasis_date_after_source_note" in result["rejected"]


@pytest.mark.parametrize(
    "field,value,prefix",
    [
        ("evidence_type", "psa_rise", "invalid_evidence_type"),
        ("assertion_type", "maybe", "invalid_assertion_type"),
        ("met_site", "pelvic_node", "invalid_met_site"),
        ("modality", "labs", "invalid_modality"),
        ("confidence", "certain", "invalid_confidence"),
    ],
)
def test_invalid_enums_are_rejected(field, value, prefix):
    result, _ = md.validate_map_result(
        {"candidates": [_candidate(**{field: value})]}, _chunk()
    )
    assert any(reason.startswith(prefix) for reason in result["rejected"])


def test_structural_problems_are_fatal_to_the_chunk():
    assert md.validate_map_result({"nope": []}, _chunk())[1] == "missing_candidates"
    assert md.validate_map_result([], _chunk())[1].startswith("non_dict_response")


def test_duplicate_candidates_are_collapsed():
    result, _ = md.validate_map_result(
        {"candidates": [_candidate(), _candidate()]}, _chunk()
    )
    assert len(result["candidates"]) == 1


def _synth(**overrides):
    payload = {
        "has_metastatic_disease": True,
        "evidence_type": "imaging_metastasis",
        "assertion_type": "established",
        "met_site": "bone",
        "metastasis_date": None,
        "source_note_date": "2021-06-03",
        "modality": "imaging",
        "supporting_quote": MET_SNIPPET,
        "confidence": "high",
        "rationale": "Imaging impression asserts osseous metastases.",
    }
    payload.update(overrides)
    return payload


def _index():
    return {(MET_SNIPPET.casefold(), "2021-06-03")}


def test_synthesis_positive_is_accepted():
    result, fatal = md.validate_synthesis_result(_synth(), _chunk(), _index())
    assert fatal is None
    assert result["has_metastatic_disease"] is True
    assert result["finding"]["met_site"] == "bone"
    assert result["finding"]["source_note_date"] == "2021-06-03"


def test_synthesis_quote_not_from_a_candidate_downgrades_to_negative():
    """Adjudication may not invent evidence; it is downgraded, never silently kept."""
    result, fatal = md.validate_synthesis_result(_synth(), _chunk(), set())
    assert fatal is None
    assert result["has_metastatic_disease"] is False
    assert result["finding"] is None
    assert result["rejected"] == {"quote_not_from_candidate": 1}


def test_synthesis_non_bool_label_is_fatal():
    result, fatal = md.validate_synthesis_result(
        _synth(has_metastatic_disease="yes"), _chunk(), _index()
    )
    assert result is None
    assert fatal.startswith("invalid_has_metastatic_disease")


def test_synthesis_negative_keeps_its_verdict():
    result, fatal = md.validate_synthesis_result(
        {
            "has_metastatic_disease": False,
            "evidence_type": None,
            "assertion_type": None,
            "met_site": None,
            "metastasis_date": None,
            "source_note_date": None,
            "modality": None,
            "supporting_quote": None,
            "rationale": "Every mention was negated.",
        },
        _chunk(),
        _index(),
    )
    assert fatal is None
    assert result["has_metastatic_disease"] is False
    assert result["rationale"] == "Every mention was negated."


def test_negative_carrying_evidence_fields_is_flagged_but_still_negative():
    result, _ = md.validate_synthesis_result(
        _synth(has_metastatic_disease=False), _chunk(), _index()
    )
    assert result["has_metastatic_disease"] is False
    assert result["rejected"] == {"negative_with_evidence_fields": 1}


# --------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------

def test_fingerprint_changes_with_model():
    a = md.extraction_run_config("scan-1", "dfci_gpt", "model-a")
    b = md.extraction_run_config("scan-1", "dfci_gpt", "model-b")
    assert a != b


def test_fingerprint_changes_with_veto_version(monkeypatch):
    """Changing the deterministic gate must force --overwrite."""
    before = md.extraction_run_config("scan-1", "dfci_gpt", "model-a")
    monkeypatch.setattr(md, "VETO_VERSION", "veto-sentinel-not-a-real-version")
    assert md.extraction_run_config("scan-1", "dfci_gpt", "model-a") != before


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def _write_evidence(tmp_path, chunks, mrn=123, cohort=(123, 999)):
    """`chunks` is a list of (chunk_index, note_date, snippet)."""
    evidence = tmp_path / "met_dx_evidence.parquet"
    pl.DataFrame(
        {
            "DFCI_MRN": [mrn] * len(chunks),
            "chunk_index": [c[0] for c in chunks],
            "note_date": [c[1] for c in chunks],
            "note_type": ["Imaging"] * len(chunks),
            "snippet": [c[2] for c in chunks],
        }
    ).write_parquet(evidence)
    write_scan_config_meta(
        tmp_path / "met_dx_evidence.meta.parquet",
        "scan-1",
        evidence_sha256=file_sha256(evidence),
        evidence_schema_version=MET_DX_EVIDENCE_SCHEMA_VERSION,
        cohort_mrns=list(cohort),
    )
    return evidence


def _one_chunk(tmp_path, snippet, note_date="2021-06-03", **kwargs):
    return _write_evidence(tmp_path, [(0, note_date, snippet)], **kwargs)


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
    """Fake provider: maps each chunk, then adjudicates from the candidates.

    The map stage echoes the note text back as the quote so grounding genuinely
    passes rather than being bypassed. The adjudication stage selects the LAST
    candidate offered, which is what makes the earliest-note-date test
    meaningful: the label date must come from the earliest qualifying
    candidate even though the adjudicator cited a later one.
    """

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
            candidate = payload["candidates"][-1]
            return json.dumps(
                {
                    "has_metastatic_disease": True,
                    "evidence_type": candidate["evidence_type"],
                    "assertion_type": candidate["assertion_type"],
                    "met_site": candidate["met_site"],
                    "metastasis_date": self.stated_date,
                    "source_note_date": candidate["source_note_date"],
                    "modality": candidate["modality"],
                    "supporting_quote": candidate["quote"],
                    "confidence": "high",
                    "rationale": "Imaging impression asserts osseous metastases.",
                }
            ), None
        note = payload["notes"][0]
        return json.dumps(
            {
                "candidates": [
                    {
                        "evidence_type": "imaging_metastasis",
                        "assertion_type": "established",
                        "met_site": "bone",
                        "stated_metastasis_date": self.stated_date,
                        "source_note_date": note["note_date"],
                        "modality": "imaging",
                        "quote": note["note_text"],
                        "confidence": "high",
                    }
                ]
            }
        ), None


def test_undated_positive_falls_back_to_the_note_date_and_is_kept(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    provider = _Provider()
    monkeypatch.setattr(md, "get_provider", lambda name: provider)
    md.run(_args(tmp_path, evidence))

    row = (
        pl.read_parquet(tmp_path / "met_dx_labels.parquet")
        .filter(pl.col("DFCI_MRN") == 123)
        .to_dicts()[0]
    )
    assert row["has_metastatic_disease"] is True
    assert row["date_source"] == "note_date"
    assert row["first_metastasis_date"] == "2021-06-03"
    assert row["met_site"] == "bone"
    assert row["label_source"] == "adjudicated"
    assert row["supporting_quote"] == MET_SNIPPET


def test_stated_date_wins_and_is_marked_stated(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    provider = _Provider(stated_date="2021-03-01")
    monkeypatch.setattr(md, "get_provider", lambda name: provider)
    md.run(_args(tmp_path, evidence))

    row = (
        pl.read_parquet(tmp_path / "met_dx_labels.parquet")
        .filter(pl.col("DFCI_MRN") == 123)
        .to_dicts()[0]
    )
    assert row["date_source"] == "stated"
    assert row["first_metastasis_date"] == "2021-03-01"


def test_earliest_qualifying_note_date_wins_over_the_cited_candidate(tmp_path, monkeypatch):
    """The note-date fallback is load-bearing, not an edge case.

    Metastatic disease is rarely given an explicit date, so most positives
    resolve here. The same fact is routinely first written by a radiologist and
    only later restated by an oncologist; dating the patient to the restatement
    would lose the original documentation, which is the answer the task is for.
    """
    evidence = _write_evidence(
        tmp_path,
        [
            (0, "2019-02-01", MET_SNIPPET),
            (1, "2021-06-03", MET_SNIPPET),
        ],
    )
    provider = _Provider()
    monkeypatch.setattr(md, "get_provider", lambda name: provider)
    md.run(_args(tmp_path, evidence))

    row = (
        pl.read_parquet(tmp_path / "met_dx_labels.parquet")
        .filter(pl.col("DFCI_MRN") == 123)
        .to_dicts()[0]
    )
    assert row["has_metastatic_disease"] is True
    assert row["first_metastasis_date"] == "2019-02-01"
    assert row["date_source"] == "note_date"
    # The quote still comes from the candidate the adjudicator actually cited.
    assert row["source_note_date"] == "2021-06-03"
    assert row["num_candidates"] == 2


def test_cohort_patients_without_evidence_become_auto_negatives(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    monkeypatch.setattr(md, "get_provider", lambda name: _Provider())
    md.run(_args(tmp_path, evidence))

    row = (
        pl.read_parquet(tmp_path / "met_dx_labels.parquet")
        .filter(pl.col("DFCI_MRN") == 999)
        .to_dicts()[0]
    )
    assert row["has_metastatic_disease"] is False
    assert row["label_source"] == "auto_negative_no_evidence"
    assert row["first_metastasis_date"] is None


def test_vetoed_evidence_yields_no_positive_and_an_audit_row(tmp_path, monkeypatch):
    """A negated note must not produce a positive, however the model answers."""
    evidence = _one_chunk(
        tmp_path, "IMPRESSION: No evidence of osseous metastatic disease."
    )
    monkeypatch.setattr(md, "get_provider", lambda name: _Provider())
    md.run(_args(tmp_path, evidence))

    labels = pl.read_parquet(tmp_path / "met_dx_labels.parquet")
    assert labels.filter(pl.col("has_metastatic_disease")).height == 0
    rejected = pl.read_parquet(tmp_path / "met_dx_rejected_findings.parquet")
    assert any(
        str(reason).startswith("negated_or_hedged:")
        for reason in rejected["reason"].to_list()
    )


def test_regional_nodal_evidence_yields_no_positive(tmp_path, monkeypatch):
    """The N1 exclusion holds end to end, not only in the unit test."""
    evidence = _one_chunk(
        tmp_path, "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes."
    )
    monkeypatch.setattr(md, "get_provider", lambda name: _Provider())
    md.run(_args(tmp_path, evidence))

    labels = pl.read_parquet(tmp_path / "met_dx_labels.parquet")
    assert labels.filter(pl.col("has_metastatic_disease")).height == 0
    rejected = pl.read_parquet(tmp_path / "met_dx_rejected_findings.parquet")
    assert "regional_nodal_only" in rejected["reason"].to_list()


def test_resume_and_config_guards(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    provider = _Provider()
    monkeypatch.setattr(md, "get_provider", lambda name: provider)
    args = _args(tmp_path, evidence)

    md.run(args)
    assert provider.calls == 2  # one map + one adjudication

    md.run(args)
    assert provider.calls == 2  # fully resumed, zero new calls

    with pytest.raises(ValueError, match="Provider, model, prompt"):
        md.run(_args(tmp_path, evidence, model="model-b"))

    monkeypatch.setattr(md, "VETO_VERSION", "veto-sentinel-not-a-real-version")
    with pytest.raises(ValueError, match="Provider, model, prompt"):
        md.run(args)


def test_evidence_content_change_is_caught(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    monkeypatch.setattr(md, "get_provider", lambda name: _Provider())
    md.run(_args(tmp_path, evidence))

    pl.DataFrame(
        {
            "DFCI_MRN": [123],
            "chunk_index": [0],
            "note_date": ["2021-06-03"],
            "note_type": ["Imaging"],
            "snippet": ["changed evidence"],
        }
    ).write_parquet(evidence)
    with pytest.raises(ValueError, match="Evidence content"):
        md.run(_args(tmp_path, evidence))


def test_rebuild_labels_only_makes_no_provider_calls(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    provider = _Provider()
    monkeypatch.setattr(md, "get_provider", lambda name: provider)
    md.run(_args(tmp_path, evidence))
    before = provider.calls

    (tmp_path / "met_dx_labels.parquet").unlink()
    md.run(_args(tmp_path, evidence, rebuild_labels_only=True))
    assert provider.calls == before
    assert pl.read_parquet(tmp_path / "met_dx_labels.parquet").height == 2


def test_rebuild_labels_only_conflicts_with_overwrite(tmp_path, monkeypatch):
    evidence = _one_chunk(tmp_path, MET_SNIPPET)
    monkeypatch.setattr(md, "get_provider", lambda name: _Provider())
    with pytest.raises(ValueError, match="cannot be combined"):
        md.run(_args(tmp_path, evidence, rebuild_labels_only=True, overwrite=True))
