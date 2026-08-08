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
from tasks.binary_NEPC.run_NEPC_classifier import make_row, validate_result
from tasks.binary_NEPC.prompts import CLASSIFY_SYSTEM_PROMPT
from tasks.cancer_stage.run_stage_extraction import validate_stage_finding
from tasks.gleason_score.build_gleason_timeline import (
    RAW_COLUMNS,
    _gleason_run_fingerprint,
    _load_patient_chunks,
    _validate_gleason_run,
    build_timeline,
    validate_gleason_finding,
)
from tasks.longitudinal_NEPC.prompts import CANONICAL_CRITERIA


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


def test_binary_avpc_prompt_uses_the_canonical_c2_contract():
    assert "C2 exclusively visceral metastases" in CLASSIFY_SYSTEM_PROMPT
    assert "Liver and other visceral organs qualify" in CLASSIFY_SYSTEM_PROMPT
    assert "concurrent bone or other non-visceral metastases mean C2 is NOT met" in (
        CLASSIFY_SYSTEM_PROMPT
    )
    assert "C2  EXCLUSIVELY visceral metastases" in CANONICAL_CRITERIA
    assert "visceral_and_bone" not in CLASSIFY_SYSTEM_PROMPT


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
        }
    ]
    valid = {
        "primary_label": "nepc",
        "has_nepc": True,
        "has_avpc": True,
        "has_biomarker": False,
        "has_molecular_avpc": False,
        "has_non_prostate_primary": False,
        "biomarker_genes": [],
        "avpc_criteria": ["C1"],
        "visceral_met_pattern": "none",
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
            "snippet": "The prostate biopsy is Gleason 4+3=7, Grade Group 3, stage III.",
        }
    ]
    stage = {
        "cancer_type": "prostate cancer",
        "staging_system": "AJCC",
        "stage_raw": "stage III",
        "stage_group": "III",
        "stage_date": None,
        "source_note_date": "2024-01-02",
        "is_historical_reference": False,
        "supporting_quote": "The prostate biopsy is Gleason 4+3=7, Grade Group 3, stage III.",
        "confidence": "high",
        "rationale": "The stage is explicit.",
    }
    assert validate_stage_finding(stage, chunk)[1] is None
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
        "quote": "The prostate biopsy is Gleason 4+3=7, Grade Group 3, stage III.",
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
