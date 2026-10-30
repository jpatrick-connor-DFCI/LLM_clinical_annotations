"""Rule-level tests for the disease-course builders: structured loaders, PSA
reconciliation and rules, therapy exposure, the statement gate, and the BCR,
progression and mCRPC criteria."""

from datetime import date, timedelta

import polars as pl
import pytest

from preprocessing.structured import (
    load_medications,
    load_psa,
    load_testosterone,
    load_vital_status,
    reduce_vital_status,
)
from preprocessing.triggers import COURSE_TRIGGER_REGEX, combined_text_pattern
from tasks.bcr.criteria import evaluate_bcr
from tasks.disease_course.psa import (
    aua_bcr,
    merge_psa,
    pcwg3_progression,
    persistent_psa,
    phoenix_bcr,
    psa_clearance,
)
from tasks.disease_course.statement_gate import screen_course_finding
from tasks.disease_course.therapy import episode_starts, exposure_at, patient_intervals, stop_points
from tasks.mcrpc.criteria import evaluate_mcrpc
from tasks.onset import both_hold
from tasks.progression.events import first_event_after, patient_events


def _labs(when, value, comparator="=", specimen=None, sid=None):
    return {"date": when, "psa_ng_ml": value, "comparator": comparator,
            "source_id": sid or f"labs:{when}:{value}", "specimen_id": specimen}


_INDEX = iter(range(10_000))


def _row(item, when=None, precision="day", quote="", **fields):
    return {
        "item": item, "event_date": when, "date_precision": precision if when else None,
        "date_source": "stated" if when else None, "quote": quote,
        "note_uid": "n", "finding_index": next(_INDEX), "source_note_date": when,
        **fields,
    }


def _note_psa(when, value, comparator="=", precision="day", quote=None):
    return _row("psa_value", when, precision, quote or f"PSA {value if value is not None else 'undetectable'}",
                psa_ng_ml=value, value_comparator=comparator)


def _day(start, days):
    return (date.fromisoformat(start) + timedelta(days=days)).isoformat()


RP = {"event_date": "2020-01-01", "date_precision": "day"}


# --- Structured loaders --------------------------------------------------------


def _write_labs(path):
    pl.DataFrame({
        "DFCI_MRN": [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "SPECIMEN_COLLECT_DT": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
                                "2020-05-01", "2020-06-01", "2020-07-01", "2020-01-01",
                                "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "TEST_TYPE_CD": ["PSA", "PSA", "PSA", "psa ", "CPSA", "PSA", "PSAULT", "PSA",
                         "TES", "TESTT", "TOTTES", "TTEST"],
        "TEST_TYPE_DESCR": ["x"] * 12,
        "NUMERIC_RESULT": [9999999.0, 9999999.0, 9999999.0, 1.5, 2.0, 20000.0, None, 4.0,
                           1.0, 20.0, 30.0, 5.0],
        "TEXT_RESULT": ["<0.1", "<", "see comment", "1.5", "2.0", "20000", ">100", "4.0",
                        "1.0", "20", "30", "5"],
        "RESULT_UOM_NM": ["ng/mL"] * 8 + ["nmol/L", "ng/dL", None, "pg/mL"],
    }).write_parquet(path)


def test_load_psa_sentinels_bounds_and_codes(tmp_path):
    _write_labs(tmp_path / "LABS.parquet")
    psa = load_psa([1], tmp_path / "LABS.parquet")
    rows = {r["date"]: r for r in psa.iter_rows(named=True)}
    assert set(rows) == {"2020-01-01", "2020-02-01", "2020-04-01", "2020-07-01"}
    assert (rows["2020-01-01"]["psa_ng_ml"], rows["2020-01-01"]["comparator"]) == (0.1, "<")
    assert rows["2020-02-01"]["psa_ng_ml"] is None
    assert rows["2020-02-01"]["comparator"] == "undetectable"
    assert (rows["2020-04-01"]["psa_ng_ml"], rows["2020-04-01"]["comparator"]) == (1.5, "=")
    assert (rows["2020-07-01"]["psa_ng_ml"], rows["2020-07-01"]["comparator"]) == (100.0, ">")
    assert all(r["source_id"].startswith("labs:") for r in rows.values())
    assert set(psa["DFCI_MRN"].to_list()) == {1}


def test_load_testosterone_units(tmp_path):
    _write_labs(tmp_path / "LABS.parquet")
    rows = {r["date"]: r for r in load_testosterone([1], tmp_path / "LABS.parquet").iter_rows(named=True)}
    assert set(rows) == {"2020-01-01", "2020-02-01", "2020-03-01"}
    assert rows["2020-01-01"]["testosterone_ng_dl"] == pytest.approx(28.84)
    assert rows["2020-02-01"]["testosterone_ng_dl"] == 20.0
    assert rows["2020-03-01"]["unit_assumed"] is True
    assert rows["2020-01-01"]["unit_assumed"] is False


def test_load_medications_classes_and_optional_fields(tmp_path):
    pl.DataFrame({
        "DFCI_MRN": [1, 1, 1, 1, 1],
        "NCI_PREFERRED_MED_NM": ["LEUPROLIDE ACETATE", "RELUGOLIX",
                                 "LUTETIUM LU 177 VIPIVOTIDE TETRAXETAN", "CARBOPLATIN",
                                 "ACETAMINOPHEN"],
        "MED_START_DT": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
        "MED_END_DT": ["2020-03-31", None, None, None, None],
    }).write_parquet(tmp_path / "MEDICATIONS.parquet")
    meds = load_medications([1], tmp_path / "MEDICATIONS.parquet")
    pairs = [(r["drug_class"], r["agent"]) for r in meds.iter_rows(named=True)]
    assert pairs == [("depot_adt", "leuprolide"), ("oral_adt", "relugolix"),
                     ("lu_psma", "lutetium_psma"), ("platinum", "carboplatin")]
    assert meds["stop_date"].to_list() == ["2020-03-31", None, None, None]
    assert meds["dose"].to_list() == [None] * 4


def test_registry_reduction_is_order_invariant_and_flags_conflicts(tmp_path):
    rows = [
        {"DFCI_MRN": 1, "death_date": "2021-05-01", "last_alive_date": "2021-06-01", "source_id": "r:0"},
        {"DFCI_MRN": 1, "death_date": "2021-04-01", "last_alive_date": "2020-01-01", "source_id": "r:1"},
        {"DFCI_MRN": 1, "death_date": "2021-05-01", "last_alive_date": None, "source_id": "r:2"},
        {"DFCI_MRN": 3, "death_date": None, "last_alive_date": "2022-01-01", "source_id": "r:3"},
    ]
    forward, backward = reduce_vital_status(rows), reduce_vital_status(rows[::-1])
    assert forward == backward
    patient = forward[0]
    assert patient["death_date"] == "2021-05-01"
    assert patient["death_date_conflict"] is True
    assert patient["death_date_candidates"] == ["2021-04-01", "2021-05-01"]
    assert patient["last_alive_date"] == "2021-06-01"
    assert patient["last_alive_after_death"] is True
    assert forward[1]["death_date"] is None and forward[1]["last_alive_after_death"] is False

    pl.DataFrame({
        "DFCI_MRN": [1, 1],
        "HYBRID_DEATH_DT": ["not a date", "2021-05-01"],
        "DERIVED_LAST_ALIVE_DATE": ["2021-01-01", "garbage"],
    }).write_parquet(tmp_path / "REG.parquet")
    loaded = load_vital_status([1], tmp_path / "REG.parquet").row(0, named=True)
    assert loaded["death_date"] == "2021-05-01" and loaded["last_alive_date"] == "2021-01-01"
    assert loaded["death_date_conflict"] is False


def test_course_trigger_regex_compiles_in_polars():
    pattern = combined_text_pattern(COURSE_TRIGGER_REGEX)
    texts = pl.Series(["PSA rose to 0.4", "s/p RALP", "mCRPC on cabazitaxel",
                       "PSMA PET negative", "making progress in PT", "no keywords"])
    hits = texts.str.contains(pattern).to_list()
    assert hits[:4] == [True, True, True, True]
    assert hits[5] is False


# --- PSA reconciliation --------------------------------------------------------


def test_note_restating_labs_is_merged_and_distinct_draw_kept():
    labs = [_labs("2021-03-01", 0.23)]
    notes = [_note_psa("2021-03-04", 0.2, quote="PSA 0.2"), _note_psa("2021-03-25", 0.4)]
    ms, audit = merge_psa(labs, notes)
    assert [(m["date"].isoformat(), m["value"], m["source"]) for m in ms] == [
        ("2021-03-01", 0.23, "labs"), ("2021-03-25", 0.4, "note"),
    ]
    assert len(ms[0]["sources"]) == 2
    assert any(a["reason"].startswith("restates:labs") for a in audit)


def test_note_rise_and_labs_confirmation_both_survive():
    ms, _ = merge_psa([_labs("2021-03-25", 0.6)], [_note_psa("2021-03-01", 0.5)])
    assert [m["source"] for m in ms] == ["note", "labs"]


def test_duplicate_note_quotes_cannot_confirm():
    rows = [_note_psa("2021-03-01", 0.3), _note_psa("2021-03-05", 0.3)]
    ms, _ = merge_psa([_labs("2020-03-01", 0.1, "<")], rows)
    assert len([m for m in ms if m["source"] == "note"]) == 1
    result = aua_bcr(ms, RP)
    assert (result["status"], result["reason"]) == ("unevaluable", "unconfirmed")


def test_same_day_conflict_is_unevaluable():
    labs = [_labs("2020-03-01", 0.1, "<"), _labs("2021-01-01", 0.5, sid="a"), _labs("2021-01-01", 3.0, sid="b")]
    ms, _ = merge_psa(labs, [])
    assert sum(m["conflict"] for m in ms) == 2
    result = aua_bcr(ms, RP)
    assert (result["status"], result["reason"]) == ("unevaluable", "conflicting_values")


def test_year_precision_note_psa_is_audit_only():
    ms, audit = merge_psa([], [_note_psa("2021-01-01", 5.0, precision="year")])
    assert ms == [] and audit[0]["reason"] == "precision_year"


# --- Persistence, clearance, AUA, Phoenix ---------------------------------------


def test_persistent_psa_without_clearance_is_not_bcr():
    ms, _ = merge_psa([_labs("2020-02-15", 0.3), _labs("2020-03-21", 0.4)], [])
    assert persistent_psa(ms, RP)["status"] is True
    result = aua_bcr(ms, RP)
    assert (result["status"], result["reason"]) == ("unevaluable", "no_clearance")


def test_later_clearance_then_confirmed_rise_dates_bcr_at_rise():
    labs = [_labs("2020-02-15", 0.3), _labs("2020-03-21", 0.4), _labs("2020-09-01", 0.05, "<"),
            _labs("2021-01-01", 0.25), _labs("2021-03-01", 0.3)]
    ms, _ = merge_psa(labs, [])
    assert persistent_psa(ms, RP)["status"] is True
    result = aua_bcr(ms, RP)
    assert result["status"] == "met" and result["date"] == "2021-01-01"
    assert result["confirmation"]["date"].isoformat() == "2021-03-01"


def test_censored_value_does_not_meet_threshold():
    labs = [_labs("2020-03-01", 0.05, "<"), _labs("2021-01-01", 0.2, "<"), _labs("2021-03-01", 0.2, "<")]
    ms, _ = merge_psa(labs, [])
    assert aua_bcr(ms, RP)["status"] == "not_met"


def test_undetectable_note_counts_as_clearance():
    notes = [_note_psa("2020-03-01", None, "undetectable", precision="month",
                       quote="PSA undetectable")]
    ms, _ = merge_psa([_labs("2021-01-01", 0.25), _labs("2021-03-01", 0.3)], notes)
    assert psa_clearance(ms, RP)["comparator"] == "undetectable"
    result = aua_bcr(ms, RP)
    assert result["status"] == "met" and result["date"] == "2021-01-01"


def test_phoenix_uses_preceding_nadir_only():
    rt = {"event_date": "2020-01-01", "date_precision": "day"}
    met, _ = merge_psa([_labs("2020-06-01", 4.0), _labs("2021-01-01", 1.0),
                        _labs("2021-06-01", 2.5), _labs("2022-01-01", 3.2)], [])
    result = phoenix_bcr(met, rt)
    assert result["status"] == "met" and result["date"] == "2022-01-01"
    assert result["nadir"]["value"] == 1.0

    future, _ = merge_psa([_labs("2020-06-01", 5.0), _labs("2021-01-01", 3.5),
                           _labs("2022-01-01", 0.5)], [])
    assert phoenix_bcr(future, rt)["status"] == "not_met"

    undetectable, _ = merge_psa([], [_note_psa("2020-06-01", None, "undetectable",
                                               quote="PSA undetectable")])
    assert phoenix_bcr(undetectable, rt)["reason"] == "no_numeric_nadir"


# --- PCWG3 ----------------------------------------------------------------------

START = {"event_date": "2021-01-01", "date_precision": "day"}


def _week(n):
    return _day("2021-01-01", 7 * n)


def test_pcwg3_baseline_branch():
    ms, _ = merge_psa([_labs("2020-12-20", 10), _labs(_week(13), 15), _labs(_week(17), 16)], [])
    result = pcwg3_progression(ms, START)
    assert result["status"] == "met" and result["date"] == _week(13)
    assert result["branch"] == "baseline"


def test_pcwg3_nadir_branch_keeps_early_nadir():
    ms, _ = merge_psa([_labs("2020-12-20", 50), _labs(_week(4), 1),
                       _labs(_week(13), 4), _labs(_week(17), 5)], [])
    result = pcwg3_progression(ms, START)
    assert result["status"] == "met" and result["date"] == _week(13)
    assert result["branch"] == "nadir" and result["nadir"]["value"] == 1


def test_pcwg3_no_baseline_and_same_day_baseline():
    ms, _ = merge_psa([_labs(_week(13), 15), _labs(_week(17), 16)], [])
    assert pcwg3_progression(ms, START)["reason"] == "no_baseline"
    same_day, _ = merge_psa([_labs("2021-01-01", 10), _labs(_week(13), 15), _labs(_week(17), 16)], [])
    assert pcwg3_progression(same_day, START)["reason"] == "no_baseline"


def test_pcwg3_early_rise_and_confirmation_outside_episode():
    early, _ = merge_psa([_labs("2020-12-20", 10), _labs(_week(4), 15), _labs(_week(8), 16)], [])
    assert pcwg3_progression(early, START)["status"] == "not_met"
    late, _ = merge_psa([_labs("2020-12-20", 10), _labs(_week(13), 15), _labs(_week(17), 16)], [])
    end = {"event_date": _week(15), "date_precision": "day"}
    result = pcwg3_progression(late, START, end)
    assert (result["status"], result["reason"]) == ("unevaluable", "unconfirmed")


def test_pcwg3_intervening_low_value_invalidates_candidate():
    ms, _ = merge_psa([_labs("2020-12-20", 10), _labs(_week(13), 15),
                       _labs(_day(_week(13), 9), 11), _labs(_week(17), 16)], [])
    result = pcwg3_progression(ms, START)
    assert result["status"] == "unevaluable"
    assert result["candidate"]["date"].isoformat() == _week(17)


# --- Therapy exposure -----------------------------------------------------------


def _med(when, agent="leuprolide", drug_class="depot_adt", **extra):
    return {"date": when, "agent": agent, "drug_class": drug_class,
            "source_id": f"medications:{agent}:{when}", **extra}


def test_single_monthly_record_is_only_estimated():
    intervals = patient_intervals([_med("2021-01-10")], [])
    may = exposure_at(intervals, date(2021, 5, 15))
    assert may["adt_active"] is None and may["adt_certainty"] == "estimated"
    assert may["therapies_possible"] == ["leuprolide"]

    formulated = patient_intervals([_med("2021-01-10", formulation="LUPRON DEPOT 7.5 MG")], [])
    assert exposure_at(formulated, date(2021, 1, 20))["adt_active"] is True
    assert exposure_at(formulated, date(2021, 5, 15))["adt_certainty"] is None


def test_repeated_depot_doses_give_supported_coverage():
    doses = [_day("2020-01-01", 90 * i) for i in range(4)]
    intervals = patient_intervals([_med(d) for d in doses], [])
    last = date.fromisoformat(doses[-1])
    assert exposure_at(intervals, last + timedelta(days=84))["adt_certainty"] == "supported"
    assert exposure_at(intervals, last + timedelta(days=105))["adt_certainty"] is None

    restart = _day(doses[-1], 400)
    with_gap = patient_intervals([_med(d) for d in doses + [restart]], [])
    assert exposure_at(with_gap, last + timedelta(days=200))["adt_certainty"] is None
    starts = episode_starts([_med(d) for d in doses + [restart]], [])
    assert [s["start"].isoformat() for s in starts] == [doses[0], restart]


def test_note_current_and_stop_statements():
    rows = [_row("therapy_current", "2021-03-01", agent="leuprolide", quote="Continues on Lupron"),
            _row("therapy_stop", "2021-04-01", agent="leuprolide", quote="Lupron stopped")]
    intervals = patient_intervals([], rows)
    stops = stop_points(rows)
    on = exposure_at(intervals, date(2021, 3, 20), stops)
    assert on["adt_active"] is True and on["adt_certainty"] == "reported"
    assert exposure_at(intervals, date(2021, 6, 1), stops)["adt_active"] is False


def test_oral_adt_days_supply():
    intervals = patient_intervals([_med("2021-01-01", "relugolix", "oral_adt", days_supply="30")], [])
    assert exposure_at(intervals, date(2021, 1, 10))["adt_active"] is True
    assert exposure_at(intervals, date(2021, 2, 15))["adt_certainty"] is None


# --- Statement gate -------------------------------------------------------------


@pytest.mark.parametrize("quote,certainty,expected", [
    ("PSMA PET: suspicious uptake in a left pelvic lymph node, concerning for recurrence.",
     "probable", (None, "probable")),
    ("PSMA PET shows recurrent disease in a 1.2 cm left obturator node.",
     "definite", (None, "definite")),
    # "consistent with" is a hedge here: the quote lowers the model's "definite".
    ("PSMA PET shows a 1.2 cm left obturator node consistent with recurrent disease.",
     "definite", (None, "probable")),
    ("No evidence of recurrent disease on PSMA PET.", "definite", "negated"),
    ("Equivocal uptake in the prostate bed on PSMA PET.", "probable", "equivocal"),
    ("Will obtain PSMA PET to evaluate for recurrence.", "probable", "hypothetical"),
])
def test_imaging_gate(quote, certainty, expected):
    reason, updates = screen_course_finding(
        _row("imaging_recurrence", "2021-05-01", quote=quote, certainty=certainty)
    )
    if isinstance(expected, tuple):
        assert reason is None and updates["certainty"] == expected[1]
    else:
        assert reason and reason.startswith(expected)


@pytest.mark.parametrize("item,quote,fields", [
    ("progression_event", "Continue enzalutamide until progression.", {"progression_type": "clinical"}),
    ("progression_event", "Stable sclerotic bone lesions, no new lesions.",
     {"progression_type": "radiographic"}),
    ("mcrpc_statement", "nmCRPC on apalutamide.", {}),
    ("radical_prostatectomy", "Prostatectomy was not performed.", {}),
])
def test_gate_rejects_non_events(item, quote, fields):
    reason, _ = screen_course_finding(_row(item, "2021-05-01", quote=quote, **fields))
    assert reason is not None


def test_gate_accepts_real_events():
    assert screen_course_finding(_row(
        "progression_event", "2021-05-01", quote="New bone lesions consistent with progression.",
        progression_type="radiographic"))[0] is None
    assert screen_course_finding(_row(
        "radical_prostatectomy", "2018-03-15", quote="s/p radical prostatectomy 3/15/2018"))[0] is None


def test_trial_context_rejects_every_item():
    row = _row("psa_value", "2021-01-01", quote="PSA 0.5", psa_ng_ml=0.5, value_comparator="=",
               _trial_context_reason="boilerplate:eligibility_context")
    assert screen_course_finding(row)[0] == "boilerplate:eligibility_context"


# --- BCR criteria ---------------------------------------------------------------


def _rp(when="2020-01-01"):
    return _row("radical_prostatectomy", when, quote="radical prostatectomy")


def test_bcr_aua_basis_and_index():
    fields = evaluate_bcr([_rp()], [_labs("2020-03-01", 0.05, "<"), _labs("2021-01-01", 0.25),
                                    _labs("2021-03-01", 0.3)])
    assert fields["bcr"] is True and fields["bcr_basis"] == "psa_aua"
    assert fields["bcr_date"] == "2021-01-01" and fields["index_date"] == "2020-01-01"
    assert fields["psa_clearance_date"] == "2020-03-01"
    assert fields["psa_bcr_confirmation_date"] == "2021-03-01"


def test_bcr_phoenix_after_primary_rt():
    rt = _row("radiation_therapy", "2019-01-01", quote="IMRT to the prostate",
              rt_intent="primary", rt_target="prostate")
    fields = evaluate_bcr([rt], [_labs("2019-06-01", 1.0), _labs("2020-06-01", 3.1)])
    assert fields["local_therapy"] == "rt" and fields["bcr_basis"] == "psa_phoenix"
    assert fields["bcr_date"] == "2020-06-01" and fields["phoenix_nadir"] == 1.0


def test_bcr_imaging_basis_and_residual_exclusion():
    image = _row("imaging_recurrence", "2021-05-01", quote="suspicious node", certainty="probable")
    cleared = evaluate_bcr([_rp(), image], [_labs("2020-03-01", 0.05, "<")])
    assert cleared["bcr_basis"] == "imaging" and cleared["bcr_date"] == "2021-05-01"
    assert cleared["imaging_certainty"] == "probable"

    persistent = evaluate_bcr([_rp(), image], [_labs("2020-02-15", 0.3), _labs("2020-03-21", 0.4)])
    assert persistent["persistent_psa"] is True and persistent["bcr"] is None
    assert persistent["imaging_exclusions"][0].endswith("residual_after_persistent_psa")


def test_bcr_persistent_statement_only_sets_flag():
    statement = _row("bcr_statement", "2020-04-01", quote="persistent PSA after RP",
                     bcr_kind="persistent_psa")
    fields = evaluate_bcr([_rp(), statement], [])
    assert fields["persistent_psa"] is True and fields["persistent_psa_basis"] == "statement"
    assert fields["bcr"] is None and fields["bcr_basis"] is None


# --- Progression events ---------------------------------------------------------


def test_progression_events_merge_statements_and_ignore_imaging_recurrence():
    rows = [
        _row("progression_event", "2021-05-01", quote="new bone lesions", progression_type="radiographic"),
        _row("progression_event", "2021-05-01", quote="progression on bone scan",
             progression_type="radiographic"),
        _row("imaging_recurrence", "2021-02-01", quote="stable node", certainty="definite"),
        _row("progression_event", "2020-06-01", quote="PSA progression", progression_type="psa"),
    ]
    vital = {"death_date": "2022-03-01", "registry_source_ids": ["registry:0"]}
    events, _ = patient_events(rows, [], [], vital)
    assert [(e["event_type"], e["event_date"]) for e in events] == [
        ("psa", "2020-06-01"), ("radiographic", "2021-05-01"), ("death", "2022-03-01"),
    ]
    assert len(events[1]["source_ids"]) == 2
    assert all(e["adt_active"] is None for e in events)

    first = first_event_after([{"DFCI_MRN": 1, **e} for e in events], {1: "2021-01-01"})
    assert first[1]["event_date"] == "2021-05-01"

    unreviewed, _ = patient_events(rows, [], [], vital, notes_reviewed=False)
    assert [e["event_type"] for e in unreviewed] == ["death"]


def test_pcwg3_events_reset_per_episode_with_exposure():
    doses = [_day("2021-01-01", 90 * i) for i in range(4)]
    labs = [_labs("2020-12-20", 10), _labs(_week(13), 15), _labs(_week(17), 16)]
    events, reasons = patient_events([], labs, [_med(d) for d in doses], None)
    psa = [e for e in events if e["basis"] == "pcwg3"]
    assert len(psa) == 1 and psa[0]["event_date"] == _week(13)
    assert psa[0]["episode_agent"] == "leuprolide" and psa[0]["adt_active"] is True
    assert psa[0]["exposure_certainty"] == "supported"


# --- mCRPC ----------------------------------------------------------------------


def _mcrpc_statement(when):
    return _row("mcrpc_statement", when, quote="Now with mCRPC.")


def test_drug_proxy_caps_later_primary():
    fields = evaluate_mcrpc([_mcrpc_statement("2021-03-01")], [],
                            [_med("2019-05-01", "cabazitaxel", "taxane")], [], None)
    assert fields["mcrpc_date"] == "2019-05-01" and fields["mcrpc_basis"] == "drug_proxy"
    assert fields["proxy_precedes_primary"] is True
    assert fields["primary_date"] == "2021-03-01" and fields["primary_basis"] == "statement"


def test_proxy_supplies_date_without_primary_but_lu_psma_does_not():
    radium = evaluate_mcrpc([], [], [_med("2022-01-01", "radium", "radium")], [], None)
    assert radium["is_mcrpc"] is True and radium["mcrpc_basis"] == "drug_proxy"
    assert radium["proxy_precedes_primary"] is False

    lu = evaluate_mcrpc([], [], [_med("2021-01-01", "lutetium_psma", "lu_psma")], [], None)
    assert lu["is_mcrpc"] is False and lu["lu_psma_date"] == "2021-01-01"

    met = {"date": "2020-06-01", "precision": "day", "source": "avpc"}
    crpc = _row("crpc_statement", "2021-02-01", quote="castration-resistant prostate cancer")
    with_crpc = evaluate_mcrpc([crpc], [], [_med("2021-01-01", "lutetium_psma", "lu_psma")], [], met)
    assert with_crpc["mcrpc_basis"] == "both_hold" and with_crpc["mcrpc_date"] == "2021-02-01"


def _derived_case(testosterone, doses=None):
    doses = doses if doses is not None else [_day("2020-01-01", 90 * i) for i in range(7)]
    labs = [_labs("2019-12-20", 10), _labs("2020-03-01", 2), _labs("2020-06-01", 5), _labs("2020-07-01", 6)]
    met = {"date": "2020-03-01", "precision": "day", "source": "avpc"}
    return evaluate_mcrpc([], labs, [_med(d) for d in doses], testosterone, met)


def _t(when, value, comparator="="):
    return {"date": when, "testosterone_ng_dl": value, "comparator": comparator, "source_id": f"t:{when}"}


def test_derived_crpc_castration_tiers():
    measured = _derived_case([_t("2020-05-15", 20)])
    assert measured["crpc_basis"] == "pcwg3" and measured["crpc_castration_tier"] == "measured"
    assert measured["mcrpc_date"] == "2020-06-01" and measured["mcrpc_basis"] == "both_hold"

    assumed = _derived_case([])
    assert assumed["crpc_castration_tier"] == "assumed"
    assert assumed["crpc_exposure_certainty"] == "supported"

    vetoed = _derived_case([_t("2020-06-10", 300)])
    assert vetoed["crpc_basis"] is None and vetoed["is_mcrpc"] is False
    assert vetoed["pcwg3_unevaluable"] == ["leuprolide@2020-01-01:testosterone_not_castrate"]

    estimated = _derived_case([], doses=["2020-01-01"])
    assert estimated["crpc_basis"] is None
    assert estimated["pcwg3_unevaluable"] == ["leuprolide@2020-01-01:adt_estimated_only"]


def test_both_hold_takes_latest_component():
    joint = both_hold([
        {"event_date": "2020-03-01", "date_precision": "day", "source_note_date": None},
        {"event_date": "2020-06-01", "date_precision": "month", "source_note_date": None},
    ])
    assert joint["event_date"] == "2020-06-01" and joint["date_precision"] == "month"
