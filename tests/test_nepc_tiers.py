import polars as pl
import pytest

from tasks.nepc import run_nepc_extraction as runner
from tasks.nepc.ne_gate import screen_ne_quote
from tasks.nepc.tiers import finding_tiers, highest_tier, patient_tiers

SMALL_CELL = ("nepc_diagnosis", "Metastatic small cell carcinoma, consistent with prostatic origin.")
TRANSFORMED = ("histologic_transformation", "biopsy showed transformation to small cell carcinoma")
CLINIC_DX = ("nepc_diagnosis", "Problem list: neuroendocrine prostate cancer, on carboplatin")
MIXED = ("mixed_ne_carcinoma", "Mixed adenocarcinoma and small cell carcinoma of the prostate")
MIXED_AS_DX = ("nepc_diagnosis", "Adenocarcinoma with a small cell carcinoma component")
PD_NE = ("pd_carcinoma_ne_features", "Poorly differentiated carcinoma with neuroendocrine features")
PD_ADENO_NE = ("ne_differentiation",
               "Prostatic adenocarcinoma, Gleason 5+5=10, with neuroendocrine differentiation")
ADENO_NE = ("ne_differentiation", "adenocarcinoma with focal neuroendocrine differentiation")
VAGUE_NE = ("ne_differentiation", "tumor shows focal neuroendocrine features")
PD = ("poorly_differentiated_carcinoma", "Poorly differentiated carcinoma, consistent with prostatic origin")
PD_ADENO = ("poorly_differentiated_carcinoma", "Poorly differentiated prostatic adenocarcinoma")
ADENO = ("prostate_adenocarcinoma", "Prostatic acinar adenocarcinoma, Gleason 4+5=9")
IHC = ("positive_ne_ihc", "Tumor cells are positive for synaptophysin and INSM1.")


def _row(finding, note="n1", date="2021-04-02", precision="day", mrn=1):
    evidence_type, quote = finding
    return {"DFCI_MRN": mrn, "note_uid": note, "evidence_type": evidence_type, "quote": quote,
            "event_date": date, "date_precision": precision, "source_note_date": date,
            "date_source": "note_date", "note_type": "Pathology"}


def _tiers(*rows):
    return {tier for tier, support in patient_tiers(list(rows)).items() if support}


@pytest.mark.parametrize("finding", [SMALL_CELL, TRANSFORMED, CLINIC_DX, MIXED, MIXED_AS_DX, PD_NE,
                                     PD_ADENO_NE, ADENO_NE, VAGUE_NE, PD, PD_ADENO, IHC])
def test_fixtures_pass_the_gate(finding):
    assert screen_ne_quote(finding[1], finding[0]) == (True, None)


@pytest.mark.parametrize("finding,expected", [
    (SMALL_CELL, {"t1"}),
    (TRANSFORMED, {"t1"}),
    (CLINIC_DX, set()),
    (MIXED, {"t1"}),
    (MIXED_AS_DX, {"t1"}),
    (PD_NE, {"t2"}),
    (PD_ADENO_NE, {"t3"}),
    (ADENO_NE, {"t3"}),
    (VAGUE_NE, set()),
    (PD, set()),
    (IHC, set()),
])
def test_finding_tiers(finding, expected):
    assert finding_tiers(_row(finding)) == expected


def test_lone_ihc_requires_histology_to_set_a_group():
    assert _tiers(_row(IHC)) == set()
    assert _tiers(_row(ADENO), _row(IHC)) == {"t3"}


def test_ihc_beside_tier1_or_tier2_is_absorbed():
    assert _tiers(_row(SMALL_CELL), _row(IHC)) == {"t1"}
    assert _tiers(_row(PD_NE), _row(IHC)) == {"t2"}


def test_pd_carcinoma_with_ihc_in_the_same_note_is_tier2():
    assert _tiers(_row(PD), _row(IHC)) == {"t2"}
    # Explicit poor differentiation sets group 2; Gleason grade alone does not.
    assert _tiers(_row(PD_ADENO), _row(IHC)) == {"t2"}
    assert _tiers(_row(ADENO), _row(IHC)) == {"t3"}


def test_pd_carcinoma_and_ihc_in_different_notes_do_not_pair():
    assert _tiers(_row(PD, note="a"), _row(IHC, note="b")) == set()


def test_pd_carcinoma_and_ihc_in_a_tier1_note_do_not_pair():
    assert _tiers(_row(SMALL_CELL), _row(PD), _row(IHC)) == {"t1"}


def test_highest_tier_precedence():
    assert highest_tier(patient_tiers([_row(ADENO_NE), _row(PD_NE, note="b")])) == "t2"
    assert highest_tier(patient_tiers([_row(ADENO_NE), _row(SMALL_CELL, note="b")])) == "t1"
    assert highest_tier(patient_tiers([_row(PD)])) is None


@pytest.mark.parametrize("quote,evidence_type,reason", [
    ("possible poorly differentiated carcinoma with neuroendocrine features",
     "pd_carcinoma_ne_features", "negated_or_hedged:possible"),
    ("high-grade neuroendocrine carcinoma", "pd_carcinoma_ne_features", "no_pd_carcinoma_term"),
    ("poorly differentiated carcinoma, no neuroendocrine features", "pd_carcinoma_ne_features",
     "negated_or_hedged:no"),
    ("no poorly differentiated carcinoma", "poorly_differentiated_carcinoma",
     "negated_or_hedged:no"),
    ("poorly differentiated carcinoma of lung origin", "poorly_differentiated_carcinoma",
     "non_prostate_primary:lung"),
    ("small cell carcinoma, extensive", "mixed_ne_carcinoma", "no_adeno_term"),
    ("adenocarcinoma; no small cell carcinoma component", "mixed_ne_carcinoma",
     "negated_or_hedged:no"),
])
def test_new_types_reject_unasserted_quotes(quote, evidence_type, reason):
    assert screen_ne_quote(quote, evidence_type) == (False, reason)


def _labels(rows, status):
    findings = pl.DataFrame(rows)
    return {r["DFCI_MRN"]: r for r in runner.build_labels(findings, status)}


def test_build_labels_tier_columns_and_unchanged_nepc_status():
    rows = [
        # Patient 1: adeno with NE diff in 2019, small cell in 2022.
        _row(ADENO_NE, note="a", date="2019-05-01", mrn=1),
        _row(SMALL_CELL, note="b", date="2022-02-01", mrn=1),
        # Patient 2: lone PD carcinoma, no NE evidence.
        _row(PD, note="c", mrn=2),
        # Patient 3: mixed tumor still counts as a stated diagnosis.
        _row(MIXED, note="d", mrn=3),
    ]
    labels = _labels(rows, {1: "ok", 2: "ok", 3: "ok", 4: "failed"})

    one = labels[1]
    assert one["nepc_status"] == "nepc" and one["nepc_date"] == "2019-05-01"
    assert (one["nepc_t1"], one["nepc_t2"], one["nepc_t3"]) == (True, False, True)
    assert one["nepc_t1_date"] == "2022-02-01" and one["nepc_t3_date"] == "2019-05-01"
    assert one["nepc_tier"] == "t1" and one["nepc_tier_date"] == "2022-02-01"

    two = labels[2]
    assert two["nepc_status"] == "non_nepc"
    assert two["evidence_types"] == [] and two["n_gated_findings"] == 0
    assert (two["nepc_t1"], two["nepc_t2"], two["nepc_t3"]) == (False, False, False)
    assert two["nepc_tier"] == "unclassified"

    three = labels[3]
    assert three["has_stated_nepc_dx"] is True
    assert three["nepc_tier"] == "t1"

    four = labels[4]
    assert four["nepc_status"] is None and four["nepc_tier"] is None and four["nepc_t1"] is None


@pytest.mark.parametrize("kind,quote,expected", [
    ("nepc_diagnosis", "Prostate biopsy: large cell neuroendocrine carcinoma", "t1"),
    ("nepc_diagnosis", "High-grade neuroendocrine carcinoma of prostatic origin", "t1"),
    ("mixed_ne_carcinoma", "Mixed prostatic adenocarcinoma and neuroendocrine carcinoma", "t1"),
    ("pd_carcinoma_ne_features", "Dedifferentiated prostatic carcinoma with neuroendocrine features", "t2"),
    ("pd_carcinoma_ne_features", "Poorly differentiated carcinoma with diffuse synaptophysin expression", "t2"),
    ("positive_ne_ihc", "Prostatic adenocarcinoma, Gleason 4+5=9, with focal weak synaptophysin staining", "t3"),
    ("ne_differentiation", "Prostatic adenocarcinoma, Grade Group 5, with diffuse neuroendocrine features", "t3"),
    ("prostate_adenocarcinoma", "Prostatic adenocarcinoma with focal neuroendocrine differentiation", "t3"),
    ("prostate_adenocarcinoma", "Prostatic acinar adenocarcinoma, Gleason 3+4=7", "t4"),
])
def test_requested_groups_through_patient_rollup(kind, quote, expected):
    row = _row((kind, quote))
    assert screen_ne_quote(quote, kind) == (True, None)
    label = _labels([row], {1: "ok"})[1]
    assert label["nepc_tier"] == expected
    assert [label[f"nepc_t{i}"] for i in range(1, 5)] == [expected == f"t{i}" for i in range(1, 5)]


@pytest.mark.parametrize("quote", [
    "Prostatic adenocarcinoma, Gleason 4+5=9",
    "Prostatic adenocarcinoma, Grade Group 5",
    "High-grade prostatic adenocarcinoma",
])
def test_grade_is_not_a_poor_differentiation_diagnosis(quote):
    assert screen_ne_quote(quote, "poorly_differentiated_carcinoma") == (False, "no_pd_carcinoma_term")


@pytest.mark.parametrize("prefix", ["No", "Possible"])
def test_unasserted_poor_differentiation_does_not_promote_group3(prefix):
    quote = f"{prefix} poorly differentiated carcinoma. Prostatic adenocarcinoma with focal neuroendocrine differentiation."
    label = _labels([_row(("ne_differentiation", quote))], {1: "ok"})[1]
    assert label["nepc_tier"] == "t3"
    assert label["nepc_t2"] is False


def test_mixed_response_does_not_downgrade_definitive_carcinoma():
    quote = "Prior adenocarcinoma. Small cell carcinoma of the prostate with mixed response to treatment."
    assert _labels([_row(("nepc_diagnosis", quote))], {1: "ok"})[1]["nepc_tier"] == "t1"


@pytest.mark.parametrize("quote", [
    "No adenocarcinoma component. Small cell carcinoma of the prostate.",
    "Possible adenocarcinoma component. Small cell carcinoma of the prostate.",
])
def test_mixed_type_requires_asserted_adenocarcinoma(quote):
    assert screen_ne_quote(quote, "mixed_ne_carcinoma")[0] is False


@pytest.mark.parametrize("quote,expected", [
    ("Liver biopsy showed small cell carcinoma of prostatic origin", "t1"),
    ("Pathology: high-grade neuroendocrine carcinoma of prostatic origin", "t1"),
    ("Biopsy-proven small cell carcinoma of the prostate", "t1"),
    ("Assessment: small cell carcinoma of the prostate", "unclassified"),
    ("Small cell carcinoma of the prostate; biopsy pending", "unclassified"),
    ("Lineage plasticity to a neuroendocrine phenotype", "unclassified"),
])
def test_clinician_requires_explicit_pathology_summary(quote, expected):
    kind = "histologic_transformation" if "plasticity" in quote else "nepc_diagnosis"
    row = {**_row((kind, quote)), "note_type": "Clinician"}
    assert _labels([row], {1: "ok"})[1]["nepc_tier"] == expected


@pytest.mark.parametrize("diagnosis,tier", [(PD, "t2"), (ADENO, "t3")])
def test_different_biopsy_dates_do_not_pair(diagnosis, tier):
    rows = [_row(diagnosis, date="2018-01-01"), _row(IHC, date="2024-01-01")]
    label = _labels(rows, {1: "ok"})[1]
    assert label["nepc_tier"] == "unclassified"
    assert label[f"nepc_{tier}"] is False


def test_combination_retains_uncertainty_when_one_component_is_undated():
    label = _labels([_row(PD, date=None, precision="unknown"), _row(IHC)], {1: "ok"})[1]
    assert label["nepc_t2"] is True
    assert label["nepc_t2_date"] is None
    assert label["nepc_t2_date_precision"] == "unknown"


def test_combination_precision_is_coarsest_overlapping_component():
    rows = [_row(PD, date="2024-01-01", precision="year"), _row(IHC, date="2024-06-02")]
    label = _labels(rows, {1: "ok"})[1]
    assert label["nepc_t2_date"] == "2024-06-02"
    assert label["nepc_t2_date_precision"] == "year"


def test_different_specimens_do_not_pair():
    rows = [{**_row(PD), "specimen_site": "prostate"},
            {**_row(IHC), "specimen_site": "liver"}]
    assert _labels(rows, {1: "ok"})[1]["nepc_tier"] == "unclassified"


def test_historical_pd_biopsy_cannot_override_current_adeno_with_ihc():
    rows = [
        _row((PD[0], "2018 biopsy showed poorly differentiated carcinoma of prostatic origin."),
             date="2018-01-01"),
        _row((ADENO[0], "2024 biopsy showed prostate adenocarcinoma."), date="2024-01-01"),
        _row((IHC[0], "2024 pathology showed focal synaptophysin positivity."), date="2024-01-01"),
    ]
    rows = [{**r, "note_type": "Clinician", "specimen_site": "prostate"} for r in rows]
    label = _labels(rows, {1: "ok"})[1]
    assert label["nepc_tier"] == "t3"
    assert label["nepc_t2"] is False
    assert label["nepc_t3_supporting_quotes"] == [rows[1]["quote"], rows[2]["quote"]]


@pytest.mark.parametrize("status", ["failed", "not_run", "partial:1/2"])
def test_incomplete_review_cannot_establish_reference_negative(status):
    label = _labels([_row(ADENO)], {1: status})[1]
    assert label["nepc_t4"] is None
    assert label["nepc_tier"] is None


def test_no_trigger_notes_are_not_reference_negatives():
    label = _labels([], {1: "no_trigger_notes"})[1]
    assert label["nepc_tier"] == "unclassified"
    assert label["nepc_t4"] is False


@pytest.mark.parametrize("extra", [SMALL_CELL, PD_NE, ADENO_NE, IHC, VAGUE_NE, CLINIC_DX])
def test_any_gated_ne_evidence_blocks_reference_group(extra):
    label = _labels([_row(ADENO), _row(extra, note="later")], {1: "ok"})[1]
    assert label["nepc_t4"] is False


def test_clinician_adenocarcinoma_alone_is_not_pathology_confirmation():
    row = {**_row(ADENO), "note_type": "Clinician"}
    assert _labels([row], {1: "ok"})[1]["nepc_tier"] == "unclassified"
    row["quote"] = "Prostate biopsy showed acinar adenocarcinoma, Gleason 4+5=9"
    assert _labels([row], {1: "ok"})[1]["nepc_tier"] == "t4"


@pytest.mark.parametrize("quote", [
    "Adenocarcinoma of unknown primary",
    "Adenocarcinoma, primary site uncertain",
    "Lung adenocarcinoma",
    "No prostatic adenocarcinoma",
])
def test_reference_histology_rejects_unestablished_or_foreign_diagnoses(quote):
    label = _labels([_row(("prostate_adenocarcinoma", quote))], {1: "ok"})[1]
    assert label["nepc_tier"] == "unclassified"
    assert label["nepc_t4"] is False


def test_ne_features_in_histology_support_preserve_broad_positive_status():
    row = _row(("prostate_adenocarcinoma", "Prostatic adenocarcinoma with focal neuroendocrine differentiation"))
    label = _labels([row], {1: "ok"})[1]
    assert label["nepc_tier"] == "t3"
    assert label["nepc_status"] == "nepc"
    assert label["nepc_t4"] is False
    row["note_type"] = "Clinician"
    label = _labels([row, _row(ADENO, note="path")], {1: "ok"})[1]
    assert label["nepc_tier"] == "unclassified"
    assert label["nepc_t4"] is False
