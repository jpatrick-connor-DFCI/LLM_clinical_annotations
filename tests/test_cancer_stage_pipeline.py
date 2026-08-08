import polars as pl
import pytest

from preprocessing.parquet_io import write_metadata
from tasks.cancer_stage.run_stage_extraction import (
    RAW_COLUMNS,
    _stage_run_fingerprint,
    _validate_stage_run,
    build_timeline,
)


def test_stage_timeline_preserves_system_and_raw_value(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    timeline_path = tmp_path / "timeline.parquet"
    pl.DataFrame(
        {
            **{column: [None] for column in RAW_COLUMNS},
            "DFCI_MRN": [1],
            "source_note_date": ["2024-01-02"],
            "cancer_type": ["ovarian cancer"],
            "histology": ["high-grade serous carcinoma"],
            "primary_site": ["left ovary"],
            "metastatic_sites": [["peritoneum", "liver"]],
            "staging_system": ["FIGO"],
            "stage_raw": ["FIGO IIIC1"],
            "stage_group": ["III"],
        }
    ).write_parquet(raw_path)

    assert build_timeline(raw_path, timeline_path) == 1
    row = pl.read_parquet(timeline_path).row(0, named=True)
    assert row["histology"] == "high-grade serous carcinoma"
    assert row["primary_site"] == "left ovary"
    assert row["metastatic_sites"] == ["liver", "peritoneum"]
    assert row["staging_system"] == "FIGO"
    assert row["stage_raw"] == "FIGO IIIC1"
    assert row["stage_group"] == "III"


def test_stage_run_fingerprint_changes_and_rejects_stale_output(tmp_path):
    evidence = tmp_path / "stage_evidence.parquet"
    pl.DataFrame({"DFCI_MRN": [1]}).write_parquet(evidence)
    first = _stage_run_fingerprint(evidence, "vertex_ai", "model", 60_000)
    second = _stage_run_fingerprint(evidence, "vertex_ai", "model", 30_000)
    assert first != second

    metadata = tmp_path / "stage_run.parquet"
    write_metadata(metadata, {"run_config": first})
    with pytest.raises(ValueError, match="changed"):
        _validate_stage_run(metadata, second, has_outputs=True, overwrite=False)
