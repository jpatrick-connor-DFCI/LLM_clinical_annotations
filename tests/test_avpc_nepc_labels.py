import polars as pl
import pytest

from tasks.longitudinal_subtype.build_nepc_timeline import TIMELINE_COLUMNS
from tasks.longitudinal_subtype import build_avpc_nepc_labels as labels

MET = "STATE:metastatic"
CR = "STATE:castration_resistant"


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


def _label(tmp_path, rows):
    timeline = tmp_path / "timeline.parquet"
    labels_path = tmp_path / "labels.parquet"
    _write_timeline(timeline, rows)
    labels.build_labels(timeline, labels_path)
    return _labels_by_mrn(labels_path)


def test_one_criterion_in_castration_resistant_metastatic_disease_is_avpc(tmp_path):
    """Aparicio Table 1: one criterion suffices, but only in mCRPC. AVPC begins
    once the criterion and both disease states are all documented."""
    row = _label(
        tmp_path,
        [
            _row(1, MET, "2020-01-01", quote="Bone metastases."),
            _row(1, "C4", "2020-04-01", quote="Bulky nodes."),
            _row(1, CR, "2020-06-01", quote="mCRPC on enzalutamide."),
        ],
    )[1]
    assert row["has_avpc"] == 1
    assert row["avpc_date"] == "2020-06-01"
    assert row["n_avpc_criteria"] == 1
    assert row["avpc_criteria"] == ["C4"]
    assert row["has_avpc_nepc"] == 1
    assert row["avpc_nepc_date"] == "2020-06-01"
    assert row["label_source"] == "timeline_positive"
    # The row that completed the definition supplies the evidence.
    assert row["supporting_quote"] == "mCRPC on enzalutamide."
    assert row["metastatic_date"] == "2020-01-01"
    assert row["castration_resistant_date"] == "2020-06-01"
    assert row["avpc_gate_missing"] == []


def test_criteria_without_the_disease_states_are_not_avpc(tmp_path):
    """Bulky nodes and an NE marker with high LDH occur outside mCRPC too."""
    row = _label(
        tmp_path,
        [_row(1, "C4", "2020-01-01"), _row(1, "C6", "2020-02-01")],
    )[1]
    assert row["has_avpc"] == 0
    assert row["avpc_date"] is None
    assert row["n_avpc_criteria"] == 2
    assert row["has_avpc_nepc"] == 0
    assert row["label_source"] == "timeline_negative"
    assert row["metastatic_date"] is None
    assert row["castration_resistant_date"] is None
    assert row["avpc_gate_missing"] == ["metastatic", "castration_resistant"]


def test_small_cell_is_exempt_from_castration_resistance_but_not_metastasis(tmp_path):
    by_mrn = _label(
        tmp_path,
        [
            _row(1, "C1", "2020-01-01"),
            _row(1, MET, "2020-03-01"),
            _row(2, "C1", "2020-01-01"),
        ],
    )
    exempt = by_mrn[1]
    assert exempt["has_avpc"] == 1
    assert exempt["avpc_date"] == "2020-03-01"
    # C1 satisfies the castration condition without being resistance.
    assert exempt["castration_resistant_date"] is None
    assert exempt["avpc_gate_missing"] == []

    localized = by_mrn[2]
    assert localized["has_avpc"] == 0
    assert localized["avpc_gate_missing"] == ["metastatic"]


@pytest.mark.parametrize(
    ("rows", "avpc_date", "metastatic_date", "resistant_date"),
    [
        # C2, C3, and C5 describe metastases.
        ([_row(1, "C2", "2020-02-01"), _row(1, CR, "2020-05-01")],
         "2020-05-01", "2020-02-01", "2020-05-01"),
        # C7 is progression to castration resistance.
        ([_row(1, MET, "2020-01-01"), _row(1, "C7", "2020-03-01")],
         "2020-03-01", "2020-01-01", "2020-03-01"),
        ([_row(1, "C3", "2020-06-01"), _row(1, "C7", "2020-02-01")],
         "2020-06-01", "2020-06-01", "2020-02-01"),
        ([_row(1, "C5", "2020-02-01"), _row(1, "C7", "2020-02-01")],
         "2020-02-01", "2020-02-01", "2020-02-01"),
    ],
)
def test_criteria_that_presuppose_a_state_supply_it(
    tmp_path, rows, avpc_date, metastatic_date, resistant_date
):
    row = _label(tmp_path, rows)[1]
    assert row["has_avpc"] == 1
    assert row["avpc_date"] == avpc_date
    assert row["metastatic_date"] == metastatic_date
    assert row["castration_resistant_date"] == resistant_date


@pytest.mark.parametrize(
    ("criterion", "missing"),
    [("C2", ["castration_resistant"]), ("C7", ["metastatic"])],
)
def test_an_implied_state_does_not_supply_the_other(tmp_path, criterion, missing):
    row = _label(tmp_path, [_row(1, criterion, "2020-01-01")])[1]
    assert row["has_avpc"] == 0
    assert row["avpc_gate_missing"] == missing


def test_same_date_tie_is_defined_by_the_first_criterion_row(tmp_path):
    row = _label(
        tmp_path,
        [
            _row(1, "C4", "2020-01-01", quote="Bulky nodes."),
            _row(1, "C2", "2020-01-01", quote="Lytic bone metastases."),
            _row(1, CR, "2020-01-01", quote="mCRPC."),
        ],
    )[1]
    assert row["has_avpc"] == 1
    assert row["avpc_date"] == "2020-01-01"
    assert row["n_avpc_criteria"] == 2
    # C2 sorts before C4, and a criterion row beats a tied disease-state row.
    assert row["supporting_quote"] == "Lytic bone metastases."


def test_nepc_precedence_overrides_timing(tmp_path):
    """AVPC met at 2020-04-10 + a NEPC feature at 2020-10-27: the union is
    timed by NEPC precedence, not by the earlier AVPC date."""
    row = _label(
        tmp_path,
        [
            _row(1, "C2", "2020-02-01"),
            _row(1, CR, "2020-04-10"),
            _row(1, "NEPC:ne_features", "2020-10-27", quote="NE features."),
        ],
    )[1]
    assert row["has_avpc"] == 1
    assert row["avpc_date"] == "2020-04-10"
    assert row["has_nepc_timeline"] == 1
    assert row["nepc_timeline_date"] == "2020-10-27"
    assert row["has_avpc_nepc"] == 1
    assert row["avpc_nepc_date"] == "2020-10-27"
    assert row["supporting_quote"] == "NE features."


def test_nepc_keys_never_contribute_to_c_count_or_disease_states(tmp_path):
    """2 C + 2 NEPC -> NEPC-positive by the NEPC rule, n_avpc_criteria == 2."""
    row = _label(
        tmp_path,
        [
            _row(1, "C4", "2020-01-01"),
            _row(1, "C6", "2020-02-01"),
            _row(1, "NEPC:small_cell_dx", "2020-03-01"),
            _row(1, "NEPC:ne_features", "2020-04-01"),
        ],
    )[1]
    assert row["has_avpc"] == 0
    assert row["n_avpc_criteria"] == 2
    assert row["avpc_gate_missing"] == ["metastatic", "castration_resistant"]
    assert row["has_nepc_timeline"] == 1
    assert row["nepc_timeline_date"] == "2020-03-01"
    assert row["has_avpc_nepc"] == 1
    assert row["avpc_nepc_date"] == "2020-03-01"
    assert row["nepc_criteria"] == ["NEPC:ne_features", "NEPC:small_cell_dx"]


def test_definition_completed_only_by_undated_evidence_is_demoted(tmp_path, capsys):
    """Undated evidence can complete the definition but cannot time it, so the
    patient is demoted rather than counted as an untimed positive."""
    undated = {"date_source": None, "date_precision": "unknown"}
    by_mrn = _label(
        tmp_path,
        [
            # Undated disease state.
            _row(1, "C4", "2020-01-01"),
            _row(1, MET, "2020-02-01"),
            _row(1, CR, None, **undated),
            # Undated criterion.
            _row(2, MET, "2020-01-01"),
            _row(2, CR, "2020-02-01"),
            _row(2, "C4", None, **undated),
        ],
    )
    for mrn, missing, n_criteria in ((1, ["castration_resistant"], 1), (2, [], 0)):
        row = by_mrn[mrn]
        assert row["has_avpc"] == 0
        assert row["avpc_date"] is None
        assert row["has_avpc_nepc"] == 0
        assert row["label_source"] == "timeline_negative"
        # Only dated criteria count toward n_avpc_criteria.
        assert row["n_avpc_criteria"] == n_criteria
        assert row["avpc_gate_missing"] == missing
    assert "definition only met via undated evidence): 2" in capsys.readouterr().out


def test_disease_states_alone_stay_conventional(tmp_path):
    """Most mCRPC is conventional: the states are not AVPC/NEPC criteria."""
    row = _label(
        tmp_path,
        [
            _row(1, MET, "2020-01-01"),
            _row(1, CR, "2020-02-01"),
            _conventional_row(1),
        ],
    )[1]
    assert row["has_avpc"] == 0
    assert row["has_avpc_nepc"] == 0
    assert row["n_avpc_criteria"] == 0
    assert row["label_source"] == "conventional"
    assert row["metastatic_date"] == "2020-01-01"
    assert row["castration_resistant_date"] == "2020-02-01"
    assert row["avpc_gate_missing"] == []


def test_conventional_row_stays_negative(tmp_path):
    row = _label(tmp_path, [_conventional_row(1)])[1]
    assert row["has_avpc"] == 0
    assert row["has_nepc_timeline"] == 0
    assert row["has_avpc_nepc"] == 0
    assert row["label_source"] == "conventional"
    assert row["n_avpc_criteria"] == 0
    assert row["metastatic_date"] is None
    assert row["castration_resistant_date"] is None
    assert row["avpc_gate_missing"] == []


def test_label_columns_only_append_gate_fields():
    """COMPASS reads these labels by column name; the gate fields go last."""
    assert labels.LABEL_COLUMNS == [
        "DFCI_MRN",
        "has_avpc",
        "avpc_date",
        "has_nepc_timeline",
        "nepc_timeline_date",
        "has_avpc_nepc",
        "avpc_nepc_date",
        "date_source",
        "date_precision",
        "n_avpc_criteria",
        "avpc_criteria",
        "nepc_criteria",
        "supporting_quote",
        "confidence",
        "source_note_date",
        "label_source",
        "metastatic_date",
        "castration_resistant_date",
        "avpc_gate_missing",
    ]


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
