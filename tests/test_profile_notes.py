from pathlib import Path
import sys
from types import SimpleNamespace

import polars as pl

from preprocessing.config import (
    DEFAULT_ADT_MRN_CSV,
    PROFILE_PATH_IMAGE_COLUMNS,
    PROFILE_PROGRESS_COLUMNS,
)
from preprocessing.cli.compile_patient_snippets import parse_args as parse_snippet_args
from preprocessing.notes import load_profile_note_mrns, load_profile_notes, load_selected_mrns
from preprocessing.cli.extract_stage_notes import (
    STAGE_TRIGGER_REGEX,
    _load_and_scan_sequential,
)
from preprocessing.triggers import combined_text_pattern
from preprocessing.cli.collect_nepc_notes import TRIGGER_REGEX as NEPC_TRIGGER_REGEX
from preprocessing.triggers import find_trigger_matches
from tasks.cancer_stage.run_stage_extraction import _normalize_stage_group


def test_profile_native_column_contract_matches_emitted_parquets():
    assert PROFILE_PATH_IMAGE_COLUMNS == (
        "RPT_ID",
        "DFCI_MRN",
        "EVENT_DATE",
        "PROC_DESC",
        "RPT_TYPE",
        "RPT_TEXT",
        "FILE",
    )
    assert PROFILE_PROGRESS_COLUMNS == (
        "RPT_ID",
        "DFCI_MRN",
        "EVENT_DATE",
        "INP_RPT_TYPE",
        "PROVIDER_TYPE",
        "ENCOUNTER_TYPE_DESC",
        "RPT_TEXT",
        "FILE",
    )


def test_mrn_cohort_list_is_read_from_csv(tmp_path):
    cohort_path = tmp_path / "prostate_mrns.csv"
    pl.DataFrame({"DFCI_MRN": [101, 202, None]}).write_csv(cohort_path)

    assert load_selected_mrns(mrn_file=cohort_path) == {101, 202}


def test_prostate_collectors_default_to_compass_profile_adt_cohort(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compile_patient_snippets.py"])

    assert parse_snippet_args().mrn_file == DEFAULT_ADT_MRN_CSV
    assert DEFAULT_ADT_MRN_CSV.name == "adt_mrns.csv"
    assert DEFAULT_ADT_MRN_CSV.parent.name == "mrn_lists"
    assert DEFAULT_ADT_MRN_CSV.parents[1].name == "COMPASS"


def _write_profile_note_fixtures(root: Path):
    pl.DataFrame(
        {
            "RPT_ID": [1, 2],
            "DFCI_MRN": [101, 202],
            "EVENT_DATE": ["2024-01-02", "2024-01-03"],
            "PROC_DESC": ["biopsy", "resection"],
            "RPT_TYPE": ["Surgical", "Surgical"],
            "RPT_TEXT": ["Stage IV disease", "No staging signal"],
            "FILE": ["pull/path-1.json", "pull/path-2.json"],
        }
    ).write_parquet(root / "PATHOLOGY_NOTES.parquet")
    pl.DataFrame(
        {
            "RPT_ID": [3],
            "DFCI_MRN": [101],
            "EVENT_DATE": ["2024-02-02"],
            "PROC_DESC": ["CT"],
            "RPT_TYPE": ["Imaging"],
            "RPT_TEXT": ["Stage two"],
            "FILE": ["pull/image-3.json"],
        }
    ).write_parquet(root / "IMAGING_NOTES.parquet")
    pl.DataFrame(
        {
            "RPT_ID": [4],
            "DFCI_MRN": [303],
            "EVENT_DATE": ["2024-03-02"],
            "INP_RPT_TYPE": ["Progress"],
            "PROVIDER_TYPE": ["MD"],
            "ENCOUNTER_TYPE_DESC": ["Clinic"],
            "RPT_TEXT": ["Clinical stage III"],
            "FILE": ["pull/progress-4.json"],
        }
    ).write_parquet(root / "PROGRESS_NOTES.parquet")


def test_load_profile_notes_standardizes_all_note_types(tmp_path):
    _write_profile_note_fixtures(tmp_path)

    notes = load_profile_notes(sorted(tmp_path.glob("*.parquet")))

    assert notes.height == 4
    assert set(notes["NOTE_TYPE"]) == {"Pathology", "Imaging", "Clinician"}
    assert (
        notes.filter(pl.col("RAW_NOTE_ID") == "1")["CLINICAL_TEXT"].item()
        == "Stage IV disease"
    )
    assert notes.filter(pl.col("RAW_NOTE_ID") == "1")["RAW_SOURCE_FILE"].item() == (
        "pull/path-1.json"
    )
    assert notes.filter(pl.col("RAW_NOTE_ID") == "4")["RPT_TYPE"].item() == "Progress"


def test_load_profile_notes_pushes_down_mrn_and_text_filters(tmp_path):
    _write_profile_note_fixtures(tmp_path)
    paths = sorted(tmp_path.glob("*.parquet"))

    by_mrn = load_profile_notes(paths, selected_mrns={101})
    stage_candidates = load_profile_notes(
        paths, text_pattern=r"(?i)(?:clinical\s+)?stage\s+(?:IV|III|two)"
    )

    assert by_mrn.height == 2
    assert set(by_mrn["DFCI_MRN"]) == {101}
    assert stage_candidates.height == 3
    assert set(stage_candidates["DFCI_MRN"]) == {101, 303}


def test_stage_scan_uses_all_profile_patients_without_an_mrn_filter(tmp_path):
    _write_profile_note_fixtures(tmp_path)
    args = SimpleNamespace(
        notes_parquet=sorted(tmp_path.glob("*.parquet")),
        note_bundle_path=None,
    )

    records = _load_and_scan_sequential(args, selected_mrns=None, context_chars=100)

    assert {record["DFCI_MRN"] for record in records} == {101, 303}


def test_profile_query_filters_note_type_before_collection(tmp_path):
    _write_profile_note_fixtures(tmp_path)

    notes = load_profile_notes(
        sorted(tmp_path.glob("*.parquet")),
        note_types=["Pathology"],
    )

    assert notes.height == 2
    assert set(notes["NOTE_TYPE"]) == {"Pathology"}


def test_profile_cohort_scan_reads_patients_independently_of_text_candidates(tmp_path):
    _write_profile_note_fixtures(tmp_path)
    paths = sorted(tmp_path.glob("*.parquet"))

    cohort = load_profile_note_mrns(paths, selected_mrns={101, 999})
    candidates = load_profile_notes(
        paths,
        selected_mrns={101, 999},
        text_pattern=r"(?i)neuroendocrine",
    )

    assert cohort == {101}
    assert candidates.is_empty()


def test_pan_cancer_stage_candidates_include_numeric_substage_and_tnm(tmp_path):
    pl.DataFrame(
        {
            "RPT_ID": [1, 2, 3],
            "DFCI_MRN": [1, 2, 3],
            "EVENT_DATE": ["2024-01-01"] * 3,
            "INP_RPT_TYPE": ["Progress"] * 3,
            "PROVIDER_TYPE": ["MD"] * 3,
            "ENCOUNTER_TYPE_DESC": ["Clinic"] * 3,
            "RPT_TEXT": ["AJCC stage 4B", "FIGO stage IIIC1", "Pathology pT3N1M0"],
            "FILE": ["pull/progress.json"] * 3,
        }
    ).write_parquet(tmp_path / "PROGRESS_NOTES.parquet")

    notes = load_profile_notes(
        [tmp_path / "PROGRESS_NOTES.parquet"],
        text_pattern=combined_text_pattern(STAGE_TRIGGER_REGEX),
    )

    assert notes.height == 3
    assert _normalize_stage_group("4B") == "IV"
    assert _normalize_stage_group("IIIC1") == "III"
    assert _normalize_stage_group("pT3N1M0") is None


def test_longitudinal_nepc_candidates_include_composite_atomic_facts():
    assert find_trigger_matches("Gleason 4+4=8", NEPC_TRIGGER_REGEX)
    assert find_trigger_matches(
        "There is no evidence of osseous metastatic disease.",
        NEPC_TRIGGER_REGEX,
    )
