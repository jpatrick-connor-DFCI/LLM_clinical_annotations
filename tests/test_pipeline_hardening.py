import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli.collect_gleason_notes import run as collect_gleason
from preprocessing.cli.collect_nepc_notes import run as collect_nepc
from preprocessing.cli.compile_patient_snippets import run as compile_snippets
from preprocessing.bundles.snippet_bundle import (
    BUNDLE_COLUMNS,
    SNIPPET_BUNDLE_FORMAT,
    load_snippet_bundle,
)
from preprocessing.notes import load_notes
from preprocessing.parquet_io import write_metadata
from preprocessing.triggers import TRIGGER_REGEX, find_trigger_matches
from tasks.binary_subtype.run_NEPC_classifier import (
    _normalize_model_result,
    classify_patient,
    make_row,
    validate_result,
)
from tasks.binary_subtype.prompts import CLASSIFY_SYSTEM_PROMPT
from tasks.cancer_stage.run_stage_extraction import validate_stage_finding
from tasks.gleason_score.build_gleason_timeline import (
    RAW_COLUMNS,
    _gleason_run_fingerprint,
    _load_patient_chunks,
    _validate_gleason_run,
    build_timeline,
    validate_gleason_finding,
)
from tasks.longitudinal_subtype.prompts import CANONICAL_CRITERIA


def test_prompt_required_nepc_and_molecular_avpc_terms_are_triggered():
    assert find_trigger_matches("Tumor cells are positive for INSM1.", TRIGGER_REGEX)
    assert find_trigger_matches(
        "Somatic PTEN loss, TP53 mutation, and RB1 deletion.", TRIGGER_REGEX
    )


def test_legacy_binary_snippet_bundle_is_rejected_after_trigger_expansion(tmp_path):
    path = tmp_path / "legacy.parquet"
    row = {column: None for column in BUNDLE_COLUMNS}
    row.update(
        {
            "bundle_format": SNIPPET_BUNDLE_FORMAT,
            "bundle_version": 2,
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata_json": "{}",
            "DFCI_MRN": 1,
            "cohort_status": "no_trigger",
        }
    )
    pl.DataFrame([row]).write_parquet(path)
    with pytest.raises(ValueError, match="Unsupported patient snippet bundle version"):
        load_snippet_bundle(path)


def test_binary_avpc_prompt_uses_the_canonical_c2_c3_contract():
    assert "C2 predominantly lytic bone metastases, rather than blastic" in (
        CLASSIFY_SYSTEM_PROMPT
    )
    assert "C3 exclusively visceral metastases" in CLASSIFY_SYSTEM_PROMPT
    assert "Liver and other visceral organs qualify" in CLASSIFY_SYSTEM_PROMPT
    assert "concurrent bone or other non-visceral metastases mean C3 is NOT met" in (
        CLASSIFY_SYSTEM_PROMPT
    )
    assert "C2  Radiographically predominantly LYTIC bone metastases" in (
        CANONICAL_CRITERIA
    )
    assert "C3  EXCLUSIVELY visceral metastases" in CANONICAL_CRITERIA
    assert "visceral_and_bone" not in CLASSIFY_SYSTEM_PROMPT


def test_binary_avpc_prompt_requires_castration_resistant_metastatic_disease():
    assert "castration-resistant, metastatic prostate" in CLASSIFY_SYSTEM_PROMPT
    assert "AVPC is castration-resistant, metastatic prostate" in CANONICAL_CRITERIA
    for field in ("has_metastatic_disease", "has_castration_resistance"):
        assert f'"{field}": true | false' in CLASSIFY_SYSTEM_PROMPT


@pytest.mark.parametrize("collector", ["binary", "gleason", "nepc"])
def test_prostate_parquet_collectors_require_an_explicit_cohort(tmp_path, collector):
    common = {
        "output_dir": tmp_path,
        "overwrite": False,
        "mrns": None,
        "mrn_file": None,
        "notes_parquet": [tmp_path / "PROGRESS_NOTES.parquet"],
        "note_bundle_path": None,
    }
    if collector == "binary":
        args = SimpleNamespace(
            **common,
            output_path=tmp_path / "snippets.parquet",
            max_notes_per_patient=75,
            scan_workers=1,
        )
        target = compile_snippets
    elif collector == "gleason":
        args = SimpleNamespace(
            **common,
            note_types=None,
            context_chars=600,
            payload_max_chars=60_000,
            scan_workers=1,
        )
        target = collect_gleason
    else:
        args = SimpleNamespace(
            **common,
            note_types=None,
            context_chars=2000,
            payload_max_chars=60_000,
            scan_workers=1,
        )
        target = collect_nepc

    with pytest.raises(ValueError, match="prostate-specific"):
        target(args)


def test_missing_explicit_bundle_does_not_fall_back_to_default_parquets(tmp_path):
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError, match="Explicit note bundle"):
        load_notes(bundle_path=missing)


def test_gleason_fingerprint_changes_with_model_and_rejects_mixed_resume(tmp_path):
    evidence = tmp_path / "gleason_evidence.parquet"
    pl.DataFrame({"DFCI_MRN": [1]}).write_parquet(evidence)
    first = _gleason_run_fingerprint(evidence, "vertex_ai", "model-a")
    second = _gleason_run_fingerprint(evidence, "vertex_ai", "model-b")
    assert first != second

    metadata = tmp_path / "gleason_run.parquet"
    write_metadata(metadata, {"run_config": first})
    with pytest.raises(ValueError, match="avoid mixed outputs"):
        _validate_gleason_run(metadata, second, has_outputs=True, overwrite=False)


def test_grade_group_only_finding_is_preserved_in_timeline(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    timeline_path = tmp_path / "timeline.parquet"
    row = {column: None for column in RAW_COLUMNS}
    row.update(
        {
            "DFCI_MRN": 1,
            "source_note_date": "2024-01-01",
            "grade_group": 4,
            "specimen_type": "biopsy",
            "quote": "Prostatic adenocarcinoma, Grade Group 4.",
        }
    )
    pl.DataFrame({column: [row[column]] for column in RAW_COLUMNS}).write_parquet(raw_path)

    assert build_timeline(raw_path, timeline_path) == 1
    output = pl.read_parquet(timeline_path)
    assert output["grade_group"].item() == 4
    assert output["gleason_total"].item() is None


def test_distinct_grade_group_only_findings_do_not_deduplicate(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    timeline_path = tmp_path / "timeline.parquet"
    rows = []
    for grade_group in (3, 4):
        row = {column: None for column in RAW_COLUMNS}
        row.update(
            {
                "DFCI_MRN": 1,
                "source_note_date": "2024-01-01",
                "grade_group": grade_group,
                "specimen_type": "biopsy",
                "quote": f"Grade Group {grade_group}.",
            }
        )
        rows.append(row)
    pl.DataFrame(
        {column: [row[column] for row in rows] for column in RAW_COLUMNS}
    ).write_parquet(raw_path)

    assert build_timeline(raw_path, timeline_path) == 2


def test_binary_result_validation_rejects_malformed_and_ungrounded_output():
    snippets = [
        {
            "note_date": "2024-01-01",
            "note_type": "Pathology",
            "trigger_categories": ["nepc"],
            "snippet": "The prostate biopsy shows small-cell neuroendocrine carcinoma.",
        },
        {
            "note_date": "2024-02-01",
            "note_type": "Imaging",
            "trigger_categories": ["avpc"],
            "snippet": "Bone scan shows widespread osseous metastases.",
        },
        {
            "note_date": "2023-06-01",
            "note_type": "Clinician",
            "trigger_categories": ["avpc"],
            "snippet": "No evidence of metastatic disease.",
        },
    ]
    valid = {
        "primary_label": "nepc",
        "has_nepc": True,
        # Small-cell histology needs metastatic disease but not castration
        # resistance (Aparicio Table 1, footnote a).
        "has_avpc": True,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": [],
        "avpc_criteria": ["C1"],
        "visceral_met_pattern": "none",
        "has_metastatic_disease": True,
        "metastatic_quote": "Bone scan shows widespread osseous metastases.",
        "metastatic_quote_date": "2024-02-01",
        "has_castration_resistance": False,
        "castration_resistance_quote": None,
        "castration_resistance_quote_date": None,
        "non_prostate_primary_types": [],
        "supporting_quotes": [
            "The prostate biopsy shows small-cell neuroendocrine carcinoma."
        ],
        "supporting_quote_dates": ["2024-01-01"],
        "confidence": "high",
        "rationale": "Pathology is definitive.",
    }
    assert validate_result(valid, snippets) == (valid, None)

    malformed = dict(valid)
    malformed.pop("has_nepc")
    assert validate_result(malformed, snippets)[1] == "invalid_boolean:has_nepc"

    invented = dict(valid)
    invented["supporting_quotes"] = ["This quote was never in the evidence."]
    assert validate_result(invented, snippets)[1] == (
        "supporting_quote_or_date_not_in_evidence"
    )

    no_metastasis = {
        "has_metastatic_disease": False,
        "metastatic_quote": None,
        "metastatic_quote_date": None,
    }
    for update, error in (
        ({"has_castration_resistance": None}, "invalid_boolean:has_castration_resistance"),
        # Without metastatic disease, small-cell histology is not AVPC.
        (no_metastasis, "inconsistent_has_avpc"),
        (
            {"has_avpc": False, "has_metastatic_disease": False},
            "state_quote_without_state:has_metastatic_disease",
        ),
        ({"metastatic_quote": " "}, "missing_state_quote:has_metastatic_disease"),
        (
            {"metastatic_quote": "Innumerable bone metastases."},
            "state_quote_or_date_not_in_evidence:has_metastatic_disease",
        ),
        (
            {"metastatic_quote_date": "2024-01-01"},
            "state_quote_or_date_not_in_evidence:has_metastatic_disease",
        ),
        (
            {
                "metastatic_quote": "No evidence of metastatic disease.",
                "metastatic_quote_date": "2023-06-01",
            },
            "state_quote_fails_gate:has_metastatic_disease:negated_or_hedged:no",
        ),
    ):
        assert validate_result(dict(valid, **update), snippets)[1] == error


def test_binary_result_normalization_repairs_deterministic_model_mistakes():
    quote = "Somatic testing found PTEN and TP53 alterations in the prostate tumor."
    state_quote = (
        "Metastatic castration-resistant prostate cancer with lytic bone metastases."
    )
    snippets = [
        {
            "note_date": "2024-03-04",
            "note_type": "Pathology",
            "trigger_categories": ["avpc", "biomarker"],
            "snippet": f"Aggressive variant prostate cancer. {state_quote} {quote}",
        }
    ]
    malformed = {
        "primary_label": "conventional",
        "has_nepc": False,
        "has_avpc": False,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": ["PTEN", "TP53"],
        "avpc_criteria": "C2",
        "visceral_met_pattern": "visceral_only",
        "has_metastatic_disease": True,
        "metastatic_quote": state_quote,
        "metastatic_quote_date": "2024-01-01",
        "has_castration_resistance": True,
        "castration_resistance_quote": state_quote,
        "castration_resistance_quote_date": None,
        "non_prostate_primary_types": [],
        "supporting_quotes": quote,
        "supporting_quote_dates": [],
        "confidence": "high",
        "rationale": "The chart documents AVPC and two molecular AVPC genes.",
    }

    normalized = _normalize_model_result(malformed, snippets)

    # C2 with documented castration resistance meets the AVPC definition.
    assert normalized["has_avpc"] is True
    assert normalized["primary_label"] == "avpc"
    assert normalized["has_molecular_avpc"] is True
    assert normalized["avpc_criteria"] == ["C2"]
    assert normalized["visceral_met_pattern"] == "none"
    assert normalized["supporting_quotes"] == [quote]
    assert normalized["supporting_quote_dates"] == ["2024-03-04"]
    assert normalized["metastatic_quote_date"] == "2024-03-04"
    assert normalized["castration_resistance_quote_date"] == "2024-03-04"
    assert normalized["avpc_gate_reasons"] == []
    assert validate_result(normalized, snippets) == (normalized, None)


def test_binary_state_quote_failing_the_mcrpc_gate_withdraws_the_state():
    lytic = "Bone scan shows predominantly lytic osseous metastases."
    trial = "Eligible for the phase 3 trial in mCRPC."
    snippets = [
        {
            "note_date": "2024-02-01",
            "note_type": "Imaging",
            "trigger_categories": ["avpc"],
            "snippet": lytic,
        },
        {
            "note_date": "2024-03-01",
            "note_type": "Clinician",
            "trigger_categories": ["avpc"],
            "snippet": trial,
        },
    ]
    result = {
        "primary_label": "avpc",
        "has_nepc": False,
        "has_avpc": True,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": [],
        "avpc_criteria": ["C2"],
        "visceral_met_pattern": "none",
        "has_metastatic_disease": True,
        "metastatic_quote": lytic,
        "metastatic_quote_date": "2024-02-01",
        "has_castration_resistance": True,
        "castration_resistance_quote": trial,
        "castration_resistance_quote_date": "2024-03-01",
        "non_prostate_primary_types": [],
        "supporting_quotes": [lytic],
        "supporting_quote_dates": ["2024-02-01"],
        "confidence": "high",
        "rationale": "Lytic bone metastases in castration-resistant disease.",
    }

    normalized = _normalize_model_result(result, snippets)

    # Trial eligibility text does not document castration resistance, so C2
    # alone falls short of the definition.
    assert normalized["has_castration_resistance"] is False
    assert normalized["castration_resistance_quote"] is None
    assert normalized["castration_resistance_quote_date"] is None
    assert normalized["has_metastatic_disease"] is True
    assert normalized["has_avpc"] is False
    assert normalized["primary_label"] == "conventional"
    assert normalized["avpc_gate_reasons"] == [
        "state_quote_rejected:castration_resistant:boilerplate:eligible",
        "missing_state:castration_resistant",
    ]
    assert validate_result(normalized, snippets) == (normalized, None)
    row = make_row(1, len(snippets), normalized)
    assert row["avpc_criteria"] == ["C2"]
    assert row["has_castration_resistance"] is False
    assert row["avpc_gate_reasons"] == normalized["avpc_gate_reasons"]

    # An ungrounded state quote is not withdrawn: validation rejects it, so the
    # model gets a correction retry.
    ungrounded = dict(result, castration_resistance_quote="Castration-resistant since 2019.")
    normalized = _normalize_model_result(ungrounded, snippets)
    assert normalized["has_castration_resistance"] is True
    assert normalized["has_avpc"] is True
    assert normalized["avpc_gate_reasons"] == []
    assert validate_result(normalized, snippets)[1] == (
        "state_quote_or_date_not_in_evidence:has_castration_resistance"
    )


def test_binary_classifier_retries_ungrounded_and_truncated_output_with_feedback():
    quote = "The prostate biopsy shows small-cell neuroendocrine carcinoma."
    snippets = [
        {
            "note_date": "2024-01-01",
            "note_type": "Pathology",
            "trigger_categories": ["nepc"],
            "snippet": quote,
        }
    ]
    valid = {
        "primary_label": "nepc",
        "has_nepc": True,
        # Small-cell histology without documented metastases is not AVPC.
        "has_avpc": False,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": [],
        "avpc_criteria": ["C1"],
        "visceral_met_pattern": "none",
        "has_metastatic_disease": False,
        "metastatic_quote": None,
        "metastatic_quote_date": None,
        "has_castration_resistance": False,
        "castration_resistance_quote": None,
        "castration_resistance_quote_date": None,
        "non_prostate_primary_types": [],
        "supporting_quotes": [quote],
        "supporting_quote_dates": ["2024-01-01"],
        "confidence": "high",
        "rationale": "Pathology is definitive.",
    }

    class Provider:
        def __init__(self):
            self.calls = []

        def call_with_retry(self, client, model, messages, max_retries):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return None, "truncated_response: finish_reason=MAX_TOKENS"
            if len(self.calls) == 2:
                ungrounded = dict(valid, supporting_quotes=["Invented quote."])
                return json.dumps(ungrounded), None
            return json.dumps(valid), None

    provider = Provider()
    result, error = classify_patient(
        provider, None, "model", 3, 1, snippets, max_output_correction_retries=2
    )

    assert error is None
    assert result == dict(valid, avpc_gate_reasons=["missing_state:metastatic"])
    assert len(provider.calls) == 3
    assert "truncated_response" in provider.calls[1][-1]["content"]
    assert "supporting_quote_or_date_not_in_evidence" in (
        provider.calls[2][-1]["content"]
    )


def test_only_brca1_or_brca2_assigns_the_biomarker_primary_label():
    snippets = [
        {
            "note_date": "2024-01-01",
            "note_type": "Pathology",
            "trigger_categories": ["biomarker"],
            "snippet": "Tumor sequencing identified a somatic PALB2 alteration.",
        }
    ]
    palb2 = {
        "primary_label": "conventional",
        "has_nepc": False,
        "has_avpc": False,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": ["PALB2"],
        "avpc_criteria": [],
        "visceral_met_pattern": "none",
        "has_metastatic_disease": False,
        "metastatic_quote": None,
        "metastatic_quote_date": None,
        "has_castration_resistance": False,
        "castration_resistance_quote": None,
        "castration_resistance_quote_date": None,
        "non_prostate_primary_types": [],
        "supporting_quotes": [
            "Tumor sequencing identified a somatic PALB2 alteration."
        ],
        "supporting_quote_dates": ["2024-01-01"],
        "confidence": "high",
        "rationale": "PALB2 is recorded but does not qualify for the biomarker bucket.",
    }
    assert validate_result(palb2, snippets) == (palb2, None)

    invalid_palb2 = dict(palb2)
    invalid_palb2.update({"primary_label": "biomarker", "has_biomarker": True})
    assert validate_result(invalid_palb2, snippets)[1] == "inconsistent_has_biomarker"

    brca2 = dict(palb2)
    brca2.update(
        {
            "primary_label": "biomarker",
            "has_biomarker": True,
            "biomarker_genes": ["BRCA2", "PALB2"],
        }
    )

    brca2_snippets = [
        {
            **snippets[0],
            "snippet": "Tumor sequencing identified somatic BRCA2 and PALB2 alterations.",
        }
    ]
    brca2["supporting_quotes"] = [
        "Tumor sequencing identified somatic BRCA2 and PALB2 alterations."
    ]
    assert validate_result(brca2, brca2_snippets) == (brca2, None)


def test_binary_output_records_non_brca_alterations_as_a_deduplicated_set():
    result = {
        "primary_label": "conventional",
        "has_nepc": False,
        "has_avpc": False,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": ["PALB2", "palb2", "ATM"],
        "avpc_criteria": [],
        "visceral_met_pattern": "none",
        "non_prostate_primary_types": [],
        "supporting_quotes": [],
        "supporting_quote_dates": [],
        "confidence": "high",
        "rationale": "Non-BRCA alterations are annotations, not a primary label.",
    }
    row = make_row(1, 1, result)
    assert row["primary_label"] == "conventional"
    assert row["biomarker_genes"] == ["ATM", "PALB2"]


def test_stage_and_gleason_findings_require_grounded_quotes_and_dates():
    chunk = [
        {
            "note_date": "2024-01-02",
            "note_type": "Pathology",
            "snippet": "Prostate adenocarcinoma with bone metastasis is Gleason 4+3=7, Grade Group 3, stage III.",
        }
    ]
    stage = {
        "cancer_type": "prostate cancer",
        "histology": "adenocarcinoma",
        "primary_site": "prostate",
        "metastatic_sites": ["bone", "bone"],
        "staging_system": "AJCC",
        "stage_raw": "stage III",
        "stage_group": "III",
        "stage_date": None,
        "source_note_date": "2024-01-02",
        "is_historical_reference": False,
        "supporting_quote": "Prostate adenocarcinoma with bone metastasis is Gleason 4+3=7, Grade Group 3, stage III.",
        "confidence": "high",
        "rationale": "The stage is explicit.",
    }
    normalized_stage, error = validate_stage_finding(stage, chunk)
    assert error is None
    assert normalized_stage["histology"] == "adenocarcinoma"
    assert normalized_stage["primary_site"] == "prostate"
    assert normalized_stage["metastatic_sites"] == ["bone"]
    invalid_sites = dict(stage, metastatic_sites="bone")
    assert validate_stage_finding(invalid_sites, chunk)[1] == "invalid_metastatic_sites"
    ungrounded_histology = dict(stage, histology="small cell carcinoma")
    assert validate_stage_finding(ungrounded_histology, chunk)[1] == (
        "histology_not_in_evidence"
    )
    stage["supporting_quote"] = "Invented stage evidence."
    assert validate_stage_finding(stage, chunk)[1] == (
        "supporting_quote_or_source_date_not_in_evidence"
    )

    gleason = {
        "primary": 4,
        "secondary": 3,
        "total": 7,
        "grade_group": 3,
        "specimen_type": "biopsy",
        "scoring_date": None,
        "source_note_date": "2024-01-02",
        "is_historical_reference": False,
        "quote": "Prostate adenocarcinoma with bone metastasis is Gleason 4+3=7, Grade Group 3, stage III.",
    }
    assert validate_gleason_finding(gleason, chunk)[1] is None
    gleason["source_note_date"] = "2024-01-03"
    assert validate_gleason_finding(gleason, chunk)[1] == (
        "quote_or_source_date_not_in_evidence"
    )


def test_sparse_gleason_chunk_indices_are_preserved():
    evidence = pl.DataFrame(
        {
            "DFCI_MRN": [123, 123, 123],
            "chunk_index": ["0", "1.5", "2"],
            "note_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "note_type": ["Pathology"] * 3,
            "snippet": ["Gleason 3+3", "bad", "Gleason 4+4"],
        }
    )
    chunks = _load_patient_chunks(evidence)
    assert set(chunks[123]) == {0, 2}


@pytest.mark.parametrize("notebook", ["cancer_stage.ipynb", "gleason_score.ipynb"])
def test_notebook_evidence_overwrite_toggle_is_wired(notebook):
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads((repo_root / "notebooks" / notebook).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'if OVERWRITE_EVIDENCE:' in code
    assert 'collect_evidence_cmd.append("--overwrite")' in code
