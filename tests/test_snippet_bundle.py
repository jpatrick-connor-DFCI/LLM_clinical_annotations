from types import SimpleNamespace

import polars as pl

from preprocessing.bundles.snippet_bundle import load_snippet_bundle
from preprocessing.cli.compile_patient_snippets import run as compile_snippets


def test_snippet_bundle_preserves_requested_patients_without_notes(tmp_path):
    notes_path = tmp_path / "PROGRESS_NOTES.parquet"
    pl.DataFrame(
        {
            "RPT_ID": [1],
            "DFCI_MRN": [101],
            "EVENT_DATE": ["2024-01-01"],
            "INP_RPT_TYPE": ["Progress"],
            "PROVIDER_TYPE": ["MD"],
            "ENCOUNTER_TYPE_DESC": ["Clinic"],
            "RPT_TEXT": ["Treatment-emergent neuroendocrine prostate cancer"],
            "FILE": ["pull/progress-1.json"],
        }
    ).write_parquet(notes_path)
    output_path = tmp_path / "snippets.parquet"
    compile_snippets(
        SimpleNamespace(
            output_path=output_path,
            overwrite=False,
            mrns="101,999",
            mrn_file=None,
            notes_parquet=[notes_path],
            note_bundle_path=None,
            max_notes_per_patient=75,
            scan_workers=1,
        )
    )

    all_mrns, snippets, metadata = load_snippet_bundle(output_path)

    assert all_mrns == {101, 999}
    assert set(snippets) == {101}
    assert metadata["no_note_mrns"] == [999]

