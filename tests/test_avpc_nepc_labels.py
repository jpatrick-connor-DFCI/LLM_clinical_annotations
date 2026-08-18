import polars as pl

from tasks.longitudinal_NEPC.build_nepc_timeline import TIMELINE_COLUMNS
from tasks.longitudinal_NEPC import build_avpc_nepc_labels as labels


def _row(
    mrn,
    criterion,
    event_date,
    *,
    date_source="stated",
    date_precision="day",
    quote="support",
    confidence="high",
    source_note_date=None,
):
    return {
        "DFCI_MRN": mrn,
        "event_date": event_date,
        "date_source": date_source,
        "date_precision": date_precision,
        "criterion_added": criterion,
        "criterion_label": criterion,
        "modality": "clinical",
        "visceral_met_pattern": None,
        "cumulative_criteria": [],
        "num_criteria_to_date": 0,
        "supporting_quote": quote,
        "confidence": confidence,
        "source_note_date": source_note_date or event_date,
    }


def _conventional_row(mrn):
    return {
        "DFCI_MRN": mrn,
        "event_date": None,
        "date_source": None,
        "date_precision": "unknown",
        "criterion_added": "conventional",
        "criterion_label": "Auto-conventional: no validated AVPC/NEPC criteria",
        "modality": "automatic",
        "visceral_met_pattern": None,
        "cumulative_criteria": [],
        "num_criteria_to_date": 0,
        "supporting_quote": None,
        "confidence": None,
        "source_note_date": None,
    }


def _write_timeline(path, rows):
    from preprocessing.parquet_io import write_rows_atomic

    write_rows_atomic(path, rows, TIMELINE_COLUMNS)


def _labels_by_mrn(path):
    return {row["DFCI_MRN"]: row for row in pl.read_parquet(path).to_dicts()}


def test_third_criterion_sets_avpc_date(tmp_path):
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(1, "C1", "2020-01-01"),
        _row(1, "C2", "2020-02-01"),
        _row(1, "C3", "2020-03-01"),
    ]
    _write_timeline(timeline, rows)

    assert labels.build_labels(timeline, labels_path) == 1
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_avpc"] == 1
    assert row["avpc_date"] == "2020-03-01"
    assert row["n_avpc_criteria"] == 3
    assert row["has_avpc_nepc"] == 1
    assert row["avpc_nepc_date"] == "2020-03-01"
    assert row["label_source"] == "timeline_positive"
    assert row["avpc_criteria"] == ["C1", "C2", "C3"]


def test_two_criteria_stays_negative(tmp_path):
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(1, "C1", "2020-01-01"),
        _row(1, "C2", "2020-02-01"),
    ]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_avpc"] == 0
    assert row["avpc_date"] is None
    assert row["n_avpc_criteria"] == 2
    assert row["has_avpc_nepc"] == 0
    assert row["label_source"] == "timeline_negative"


def test_same_date_block_jumps_from_one_to_three(tmp_path):
    """A same-date block pushing the count 1 -> 3 dates the event at that block."""
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(1, "C1", "2020-01-01"),
        _row(1, "C2", "2020-06-01"),
        _row(1, "C3", "2020-06-01"),
    ]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_avpc"] == 1
    assert row["avpc_date"] == "2020-06-01"
    assert row["n_avpc_criteria"] == 3


def test_nepc_precedence_overrides_timing(tmp_path):
    """C-threshold at day 100 (2020-04-10) + NEPC feature at day 300 (2020-10-27)
    -> has_avpc==0 (per AVPC's own definition, zero NEPC criteria ever is
    required), has_nepc_timeline==1, avpc_nepc_date == the NEPC date."""
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(1, "C1", "2020-01-01"),
        _row(1, "C2", "2020-02-01"),
        _row(1, "C3", "2020-04-10"),
        _row(1, "NEPC:ne_features", "2020-10-27"),
    ]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_nepc_timeline"] == 1
    assert row["nepc_timeline_date"] == "2020-10-27"
    assert row["has_avpc_nepc"] == 1
    assert row["avpc_nepc_date"] == "2020-10-27"
    # n_avpc_criteria still reflects the C-only count reached along the way.
    assert row["n_avpc_criteria"] == 3


def test_nepc_keys_never_contribute_to_c_count(tmp_path):
    """2 C + 2 NEPC -> NEPC-positive by the NEPC rule, n_avpc_criteria == 2."""
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(1, "C1", "2020-01-01"),
        _row(1, "C2", "2020-02-01"),
        _row(1, "NEPC:small_cell_dx", "2020-03-01"),
        _row(1, "NEPC:ne_features", "2020-04-01"),
    ]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_avpc"] == 0
    assert row["n_avpc_criteria"] == 2
    assert row["has_nepc_timeline"] == 1
    assert row["nepc_timeline_date"] == "2020-03-01"
    assert row["has_avpc_nepc"] == 1
    assert row["avpc_nepc_date"] == "2020-03-01"
    assert row["nepc_criteria"] == ["NEPC:ne_features", "NEPC:small_cell_dx"]


def test_undated_only_positive_is_demoted(tmp_path):
    """A patient who would cross 3 criteria only via undated evidence is
    demoted to has_avpc_nepc = 0."""
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(1, "C1", "2020-01-01"),
        _row(1, "C2", "2020-02-01"),
        # Undated: event_date is None.
        _row(1, "C3", None, date_source=None, date_precision="unknown"),
    ]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_avpc"] == 0
    assert row["avpc_date"] is None
    assert row["has_avpc_nepc"] == 0
    # Only dated criteria count toward n_avpc_criteria.
    assert row["n_avpc_criteria"] == 2
    assert row["label_source"] == "timeline_negative"


def test_conventional_row_stays_negative(tmp_path):
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [_conventional_row(1)]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    row = _labels_by_mrn(labels_path)[1]
    assert row["has_avpc"] == 0
    assert row["has_nepc_timeline"] == 0
    assert row["has_avpc_nepc"] == 0
    assert row["label_source"] == "conventional"
    assert row["n_avpc_criteria"] == 0


def test_missing_timeline_file_is_non_fatal(tmp_path):
    timeline = tmp_path / "does_not_exist.parquet"
    labels_path = tmp_path / "labels.parquet"
    assert labels.build_labels(timeline, labels_path) == 0
    output = pl.read_parquet(labels_path)
    assert output.height == 0
    assert output.columns == labels.LABEL_COLUMNS


def test_output_is_sorted_by_mrn(tmp_path):
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    rows = [
        _row(30, "C1", "2020-01-01"),
        _row(10, "C1", "2020-01-01"),
        _row(20, "C1", "2020-01-01"),
    ]
    _write_timeline(timeline, rows)

    labels.build_labels(timeline, labels_path)
    output = pl.read_parquet(labels_path)
    assert output["DFCI_MRN"].to_list() == [10, 20, 30]
