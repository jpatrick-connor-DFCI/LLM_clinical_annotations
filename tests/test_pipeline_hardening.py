import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.cli.collect_gleason_notes import run as collect_gleason
from preprocessing.cli.collect_nepc_dx_notes import run as collect_nepc
from preprocessing.cli.compile_patient_snippets import run as compile_snippets
from preprocessing.bundles.snippet_bundle import (
    BUNDLE_COLUMNS,
    SNIPPET_BUNDLE_FORMAT,
    load_snippet_bundle,
)
from preprocessing.notes import load_notes
from preprocessing.parquet_io import write_metadata
from preprocessing.triggers import TRIGGER_REGEX, find_trigger_matches
from tasks.cancer_stage.run_stage_extraction import validate_stage_finding
from tasks.gleason_score.build_gleason_timeline import (
    RAW_COLUMNS,
    _gleason_run_fingerprint,
    _load_patient_chunks,
    _validate_gleason_run,
    build_timeline,
    validate_gleason_finding,
)


def test_prompt_required_nepc_and_molecular_avpc_terms_are_triggered():
    assert find_trigger_matches("Tumor cells are positive for INSM1.", TRIGGER_REGEX)
    assert find_trigger_matches(
        "Somatic PTEN loss, TP53 mutation, and RB1 deletion.", TRIGGER_REGEX
    )


def test_legacy_snippet_bundle_is_rejected_after_trigger_expansion(tmp_path):
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


@pytest.mark.parametrize("collector", ["snippets", "gleason", "nepc"])
def test_prostate_parquet_collectors_require_an_explicit_cohort(tmp_path, collector):
    common = {
        "output_dir": tmp_path,
        "overwrite": False,
        "mrns": None,
        "mrn_file": None,
        "notes_parquet": [tmp_path / "PROGRESS_NOTES.parquet"],
        "note_bundle_path": None,
    }
    if collector == "snippets":
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
