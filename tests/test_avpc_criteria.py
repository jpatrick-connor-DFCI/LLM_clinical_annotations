import pytest

from tasks.avpc.criteria import (
    evaluate_patient,
    gleason_in_quote,
    measurements_cm,
    screen_finding,
)


def row(item, quote, date="2021-01-15", precision="day", source="stated", **fields):
    return {
        "DFCI_MRN": 1,
        "item": item,
        "quote": quote,
        "event_date": date,
        "date_precision": precision,
        "date_source": source,
        "source_note_date": date,
        **fields,
    }


MET = row("metastatic_disease", "Metastatic prostate cancer to bone.", "2020-01-10")
CRPC = row("castration_resistance", "Now castration-resistant prostate cancer.", "2021-01-10")


@pytest.mark.parametrize("quote,expected", [
    ("There are 2 bone metastases and 25 liver metastases.", False),
    ("25 liver metastases and 2 bone metastases.", False),
    ("2 bone metastases. Liver: 25 metastases.", False),
    ("2 bone metastases and 25 metastases in the liver.", False),
    ("2 bone metastases and 25 nodal lesions.", False),
    ("25 bone metastases and 2 liver metastases.", True),
    ("Bone scan shows 25 metastatic lesions.", True),
    ("25 metastases in the skeleton.", True),
])
def test_c5_counts_only_bone_lesions(quote, expected):
    bone = row("bone_metastasis", quote, count=2 if not expected else 25)
    psa = row("low_psa", "At initial presentation before ADT, PSA was 4 ng/mL.",
              psa_ng_ml=4., psa_context="initial_presentation_pre_adt")
    fields, _ = evaluate_patient([MET, CRPC, bone, psa])
    assert fields["c5_met"] is expected
    assert fields["is_avpc"] is expected


@pytest.mark.parametrize("date,precision", [(None, "unknown"), ("2021-01-01", "year")])
def test_c4_and_avpc_preserve_uncertain_prerequisite_dates(date, precision):
    mass = row("bulky_prostate_pelvic_mass", "Prostate mass measuring 6 cm.",
               "2021-06-01", size_cm=6.)
    grade = row("gleason_high_grade", "Gleason 4+5=9.", date, precision, gleason_total=9)
    fields, _ = evaluate_patient([MET, CRPC, mass, grade])
    assert fields["is_avpc"] is True
    assert fields["c4_met"] is True
    assert fields["c4_date"] == ("2021-06-01" if date else None)
    assert fields["avpc_date"] == fields["c4_date"]
    assert fields["c4_date_precision"] == precision
    assert fields["avpc_date_precision"] == precision


def test_c6_undated_component_keeps_positive_label_with_unknown_onset():
    marker = row("ne_marker", "Synaptophysin positive.", None, "unknown",
                 specimen="tissue", marker="synaptophysin")
    lab = row("elevated_ldh", "LDH 600 (ULN 250).", value=600., uln=250.)
    fields, _ = evaluate_patient([MET, CRPC, marker, lab])
    assert fields["is_avpc"] and fields["c6_met"]
    assert fields["c6_date"] is None and fields["avpc_date"] is None
    # A separate, fully dated criterion can still establish a known onset.
    node = row("bulky_lymphadenopathy", "Pelvic lymph node measuring 6 cm.", size_cm=6.)
    fields, _ = evaluate_patient([MET, CRPC, marker, lab, node])
    assert fields["avpc_date"] == node["event_date"]


@pytest.mark.parametrize("date,precision", [(None, "unknown"), ("2021-01-01", "year")])
def test_composite_precision_survives_label_parquet(tmp_path, date, precision):
    import polars as pl
    from tasks import note_extraction
    from tasks.avpc import run_avpc_extraction as runner
    from tasks.avpc_adjacent import run_avpc_adjacent_extraction as adjacent

    rows = [MET, CRPC,
            row("bulky_prostate_pelvic_mass", "Prostate mass measuring 6 cm.",
                "2021-06-01", size_cm=6.),
            row("gleason_high_grade", "Gleason 4+5=9.", date, precision, gleason_total=9)]
    for module, prefix, date_column in [(runner, "", "avpc_date"),
                                        (adjacent, "avpc_adj_", "avpc_adjacent_date")]:
        findings = note_extraction._frame(rows, module.SPEC.findings_schema)
        labels = note_extraction._frame(module.build_labels(findings, {1: "ok"}), module.LABEL_SCHEMA)
        path = tmp_path / f"{prefix}labels.parquet"
        labels.write_parquet(path)
        label = pl.read_parquet(path).row(0, named=True)
        assert label[f"{prefix}c4_date_precision"] == precision
        assert label[f"{date_column}_precision"] == precision
        assert label[date_column] == ("2021-06-01" if date else None)


def test_size_and_gleason_parsing():
    assert measurements_cm("node measures 5.4 x 3.2 cm") == [5.4, 3.2]
    assert measurements_cm("node measures 54 mm") == [pytest.approx(5.4)]
    assert gleason_in_quote(9, "Gleason 4+5=9 adenocarcinoma")
    assert gleason_in_quote(8, "Gleason score of 8")
    assert not gleason_in_quote(9, "Gleason 4+4")


def test_hallucinated_number_not_in_quote_is_rejected():
    node = row("bulky_lymphadenopathy", "Bulky retroperitoneal adenopathy.", size_cm=6.0)
    assert screen_finding(node) == "size_not_in_quote"
    psa = row("low_psa", "PSA was low at progression.", psa_ng_ml=2.0,
              psa_context="symptomatic_crpc_progression")
    assert screen_finding(psa) == "psa_not_in_quote"


def test_mcrpc_with_c4_node_is_avpc():
    node = row("bulky_lymphadenopathy",
               "Retroperitoneal lymph node conglomerate measuring 6.2 x 4.0 cm.",
               "2021-02-01", size_cm=6.2)
    fields, _ = evaluate_patient([MET, CRPC, node])
    assert fields["is_metastatic"] and fields["is_castration_resistant"] and fields["is_mcrpc"]
    assert fields["c4_met"] and fields["c4_size_quote"] == node["quote"]
    assert fields["avpc_criteria"] == ["C4"]
    assert fields["is_avpc"] is True
    assert fields["avpc_date"] == "2021-02-01"


def test_criterion_without_crpc_is_not_avpc():
    node = row("bulky_lymphadenopathy", "Pelvic lymph node measuring 55 mm.", size_cm=5.5)
    fields, _ = evaluate_patient([MET, node])
    assert fields["c4_met"] is True
    assert fields["is_avpc"] is False


def test_c4_small_node_and_mass_without_high_grade():
    small = row("bulky_lymphadenopathy", "Lymph node measuring 3.1 cm.", size_cm=3.1)
    mass = row("bulky_prostate_pelvic_mass", "Prostate mass measuring 6.0 cm.", size_cm=6.0)
    fields, _ = evaluate_patient([MET, CRPC, small, mass])
    assert fields["c4_met"] is False
    grade = row("gleason_high_grade", "Gleason 4+5=9.", "2018-01-01", gleason_total=9)
    fields, _ = evaluate_patient([MET, CRPC, small, mass, grade])
    assert fields["c4_met"] is True
    assert fields["c4_gleason_quote"] == "Gleason 4+5=9."


@pytest.mark.parametrize("node_date", ["2020-12-01", "2023-01-01"])
def test_c4_uses_earliest_qualifying_branch(node_date):
    mass = row("bulky_prostate_pelvic_mass", "Prostate mass measuring 6.0 cm.",
               "2021-02-01", size_cm=6.0)
    grade = row("gleason_high_grade", "Gleason 4+5=9.", "2018-01-01", gleason_total=9)
    node = row("bulky_lymphadenopathy", "Retroperitoneal lymph node measuring 6.0 cm.",
               node_date, size_cm=6.0)
    fields, _ = evaluate_patient([MET, CRPC, mass, grade, node])
    expected = min(node_date, mass["event_date"])
    assert fields["c4_date"] == expected
    assert fields["avpc_date"] == max(CRPC["event_date"], expected)
    assert fields["c4_size_quote"] == (node if node_date == expected else mass)["quote"]
    assert fields["c4_gleason_quote"] == (None if node_date == expected else grade["quote"])


def test_c1_is_exempt_from_castration_resistance_but_needs_metastatic():
    small_cell = row("small_cell_histology",
                     "Liver biopsy: small cell carcinoma, prostatic origin.", "2021-03-01")
    fields, _ = evaluate_patient([small_cell])
    assert fields["c1_met"] and not fields["is_avpc"]
    fields, _ = evaluate_patient([MET, small_cell])
    assert fields["is_avpc"] is True
    assert fields["avpc_date"] == "2021-03-01"


def test_negated_small_cell_does_not_meet_c1():
    negated = row("small_cell_histology", "No small cell component identified.")
    fields, rejections = evaluate_patient([MET, CRPC, negated])
    assert fields["c1_met"] is False
    assert "small_cell_histology:negated_or_hedged:no" in rejections


def test_c2_defeated_by_earlier_bone_allowed_when_bone_is_later():
    visceral = row("visceral_metastasis",
                   "Multiple liver metastases. No osseous or nodal metastatic disease.",
                   "2021-05-01", organ="liver", exclusively_visceral=True)
    later_bone = row("bone_metastasis", "New osseous metastases in the spine.", "2022-01-01")
    fields, _ = evaluate_patient([CRPC, visceral, later_bone])
    assert fields["c2_met"] is True
    assert fields["is_metastatic"] and fields["is_avpc"]

    earlier_bone = row("bone_metastasis", "Osseous metastases in the spine.", "2020-06-01")
    fields, rejections = evaluate_patient([CRPC, visceral, earlier_bone])
    assert fields["c2_met"] is False
    assert "visceral_metastasis:c2_bone_disease_on_or_before" in rejections


def test_c2_needs_exclusivity_in_quote():
    visceral = row("visceral_metastasis", "Multiple liver metastases.",
                   organ="liver", exclusively_visceral=True)
    fields, rejections = evaluate_patient([CRPC, visceral])
    assert fields["c2_met"] is False
    assert "visceral_metastasis:c2_exclusivity_not_in_quote" in rejections


@pytest.mark.parametrize("quote", [
    "Liver metastases; no osseous metastatic disease.",
    "Liver metastases; no nodal metastatic disease.",
    "Liver metastases; no bone metastases and extensive nodal metastases.",
])
def test_c2_requires_absence_of_both_nonvisceral_sites(quote):
    visceral = row("visceral_metastasis", quote, exclusively_visceral=True)
    fields, rejections = evaluate_patient([CRPC, visceral])
    assert fields["c2_met"] is False
    assert fields["is_avpc"] is False
    assert "visceral_metastasis:c2_exclusivity_not_in_quote" in rejections


@pytest.mark.parametrize("quote", [
    "Exclusively visceral metastases in the liver.",
    "Liver metastases only.",
    "Liver metastases. No bone or nodal metastases.",
    "Liver metastases. No nodal or osseous metastases.",
    "Liver metastases. No evidence of bone metastases. No lymph node involvement.",
])
def test_c2_accepts_full_exclusivity_evidence(quote):
    visceral = row("visceral_metastasis", quote, exclusively_visceral=True)
    fields, _ = evaluate_patient([CRPC, visceral])
    assert fields["c2_met"] is True
    assert fields["is_avpc"] is True


@pytest.mark.parametrize("nodal_quote", [
    "Distant nodal metastases in para-aortic lymph nodes.",
    "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes.",
])
@pytest.mark.parametrize("nodal_date, expected", [
    ("2020-01-01", False), ("2021-05-01", False), ("2022-01-01", True),
])
def test_c2_vetoes_prior_and_concurrent_nodal_metastases(nodal_quote, nodal_date, expected):
    visceral = row("visceral_metastasis", "Exclusively liver metastases.",
                   "2021-05-01", exclusively_visceral=True)
    nodes = row("metastatic_disease", nodal_quote, nodal_date)
    fields, rejections = evaluate_patient([CRPC, visceral, nodes])
    assert fields["c2_met"] is expected
    if not expected:
        assert "visceral_metastasis:c2_nodal_disease_on_or_before" in rejections


def test_c2_does_not_treat_negated_or_reactive_nodes_as_metastases():
    visceral = row("visceral_metastasis", "Exclusively liver metastases.",
                   exclusively_visceral=True)
    negated = row("metastatic_disease", "No nodal metastases.")
    reactive = row("bulky_lymphadenopathy", "Reactive lymph node measuring 1.5 cm.", size_cm=1.5)
    assert evaluate_patient([CRPC, visceral, negated, reactive])[0]["c2_met"] is True


def test_c3_lytic_and_blastic_rejection():
    lytic = row("bone_metastasis", "Predominantly lytic osseous metastases.",
                lytic_predominant=True)
    fields, _ = evaluate_patient([CRPC, lytic])
    assert fields["c3_met"] is True
    assert fields["metastatic_source"] == "quote"

    mixed = row("bone_metastasis", "Mixed lytic and sclerotic osseous metastases.",
                lytic_predominant=True)
    fields, rejections = evaluate_patient([CRPC, mixed])
    assert fields["c3_met"] is False
    assert "bone_metastasis:c3_lytic_predominance_not_stated" in rejections


def test_c5_pairing_window():
    psa = row("low_psa", "At symptomatic progression PSA was 4.2 ng/mL.", "2021-06-01",
              psa_ng_ml=4.2, psa_context="symptomatic_crpc_progression")
    bone = row("bone_metastasis", "Bone scan: innumerable osseous metastases.", "2021-07-15")
    fields, _ = evaluate_patient([CRPC, psa, bone])
    assert fields["c5_met"] is True
    assert fields["c5_date"] == "2021-07-15"
    assert fields["c5_psa_quote"] == psa["quote"] and fields["c5_bone_quote"] == bone["quote"]

    far_bone = {**bone, "event_date": "2022-01-15", "source_note_date": "2022-01-15"}
    fields, _ = evaluate_patient([CRPC, psa, far_bone])
    assert fields["c5_met"] is False


def test_c5_needs_low_psa_and_high_volume():
    high_psa = row("low_psa", "At progression PSA was 45 ng/mL.", psa_ng_ml=45.0,
                   psa_context="symptomatic_crpc_progression")
    bone = row("bone_metastasis", "At least 24 osseous metastases.", count=24)
    assert evaluate_patient([CRPC, high_psa, bone])[0]["c5_met"] is False
    few = row("bone_metastasis", "Three osseous metastases.", count=3)
    low = {**high_psa, "quote": "At progression PSA was 4.5 ng/mL.", "psa_ng_ml": 4.5}
    assert evaluate_patient([CRPC, low, few])[0]["c5_met"] is False
    assert evaluate_patient([CRPC, low, bone])[0]["c5_met"] is True


def test_c6_uln_ratio():
    marker = row("ne_marker", "Tumor cells are positive for synaptophysin.",
                 marker="synaptophysin", specimen="tissue")
    ldh_low = row("elevated_ldh", "LDH 300 (ULN 250).", value=300.0, uln=250.0)
    ldh_high = row("elevated_ldh", "LDH 600 (ULN 250).", value=600.0, uln=250.0)
    assert evaluate_patient([MET, CRPC, marker, ldh_low])[0]["c6_met"] is False
    fields, _ = evaluate_patient([MET, CRPC, marker, ldh_high])
    assert fields["c6_met"] is True
    assert fields["c6_lab_quote"] == ldh_high["quote"]
    stated = row("elevated_cea", "CEA 3 times the upper limit of normal.")
    assert evaluate_patient([MET, CRPC, marker, stated])[0]["c6_met"] is True


def test_c6_cd56_alone_is_not_a_paper_marker():
    marker = row("ne_marker", "Tumor cells are positive for CD56.", marker="CD56",
                 specimen="tissue")
    ldh = row("elevated_ldh", "LDH 600 (ULN 250).", value=600.0, uln=250.0)
    assert evaluate_patient([MET, CRPC, marker, ldh])[0]["c6_met"] is False


def _adt(date, precision="month", source="stated"):
    return row("adt_start", "Started leuprolide.", date, precision, source, agent="leuprolide")


def _crpc(date, precision="day"):
    return row("castration_resistance", "Castration-resistant prostate cancer.", date, precision)


@pytest.mark.parametrize(
    "adt,crpc,met",
    [
        (_adt("2020-01-01"), _crpc("2020-05-20"), True),
        (_adt("2020-01-01"), _crpc("2020-06-30"), True),
        (_adt("2020-01-01"), _crpc("2020-07-05"), False),
        (_adt("2020-01-01"), _crpc("2020-01-20"), False),  # same month: not certainly after
        (_adt("2020-01-01", precision="year"), _crpc("2020-05-20"), False),
        (_adt("2020-01-01", source="note_date"), _crpc("2020-05-20"), False),
    ],
)
def test_c7_interval_boundaries_and_precision(adt, crpc, met):
    fields, _ = evaluate_patient([MET, adt, crpc])
    assert fields["c7_met"] is met
    if met:
        assert fields["c7_adt_start_quote"] == adt["quote"]
        assert fields["c7_interval_days"] > 0


def test_c7_explicit_statement_implies_crpc():
    rapid = row("rapid_progression_on_adt",
                "PSA progression within 4 months of starting ADT.", "2020-05-01")
    fields, _ = evaluate_patient([MET, rapid])
    assert fields["c7_met"] is True
    assert fields["is_castration_resistant"] is True
    assert fields["crpc_source"] == "implied:C7"
    assert fields["is_avpc"] is True

    hedged = row("rapid_progression_on_adt",
                 "Patients who progress within 6 months of ADT may have AVPC.")
    assert evaluate_patient([MET, hedged])[0]["c7_met"] is False


@pytest.mark.parametrize("quote", [
    "No progression within 6 months of starting ADT.",
    "He did not progress within 4 months of starting ADT.",
    "Progression within 4 months of starting ADT was not observed.",
    "Progression within 4 months of ADT was ruled out.",
    "Primary refractoriness was not identified; no progression within 4 months of ADT.",
    "Not primary refractory to ADT.",
    "Primary refractory to docetaxel.",
    "Never responded to docetaxel.",
    "Primary refractory to docetaxel within 4 months, while on ADT.",
    "Refractory to docetaxel within 4 months, while on ADT.",
])
def test_c7_rejects_negated_progression_and_nonhormonal_refractoriness(quote):
    rapid = row("rapid_progression_on_adt", quote)
    fields, rejections = evaluate_patient([MET, rapid])
    assert fields["c7_met"] is False
    assert fields["is_castration_resistant"] is False
    assert fields["is_avpc"] is False
    assert rejections


@pytest.mark.parametrize("quote", [
    "Primary refractory to ADT.",
    "Primarily resistant to leuprolide.",
    "Primary castration-resistant prostate cancer.",
    "Primary hormone-refractory disease.",
    "Never responded to ADT.",
    "Never had a response to leuprolide.",
])
def test_c7_preserves_asserted_primary_hormonal_refractoriness(quote):
    rapid = row("rapid_progression_on_adt", quote)
    fields, rejections = evaluate_patient([MET, rapid])
    assert fields["c7_met"] is True
    assert fields["is_avpc"] is True
    assert rejections == []


@pytest.mark.parametrize("explicit_date", ["2020-03-01", "2021-01-01"])
def test_c7_uses_earliest_explicit_or_calculated_onset(explicit_date):
    adt = _adt("2020-01-01", precision="day")
    crpc = _crpc("2020-04-01")
    rapid = row("rapid_progression_on_adt", "Primary refractory to ADT.",
                explicit_date, source="note_date")
    fields, _ = evaluate_patient([MET, adt, crpc, rapid])
    expected = min(explicit_date, crpc["event_date"])
    assert fields["c7_date"] == expected
    assert fields["avpc_date"] == expected
    assert fields["c7_interval_days"] == (None if explicit_date == expected else 91)


@pytest.mark.parametrize("item,quote,extra", [
    ("gleason_high_grade", "Eligibility criteria: Gleason score 8 or higher.", {"gleason_total": 8}),
    ("low_psa", "Eligibility criteria: PSA at diagnosis <=10 ng/mL.",
     {"psa_ng_ml": 10, "psa_context": "initial_presentation_pre_adt"}),
    ("elevated_ldh", "Eligibility criteria: LDH 2 times the upper limit of normal.", {}),
    ("elevated_cea", "Eligibility criteria: CEA 2 times the upper limit of normal.", {}),
])
def test_numeric_findings_cannot_bypass_trial_guard(item, quote, extra):
    finding = row(item, quote, **extra)
    assert screen_finding(finding) is not None


def test_cropped_eligibility_quote_uses_source_context_guard():
    finding = row("elevated_ldh", "LDH 2 times the upper limit of normal.",
                  _trial_context_reason="boilerplate:eligibility_context")
    assert screen_finding(finding) == "boilerplate:eligibility_context"


def test_regional_nodes_alone_are_not_metastatic():
    nodes = row("metastatic_disease",
                "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes.")
    fields, rejections = evaluate_patient([CRPC, nodes])
    assert fields["is_metastatic"] is False
    assert "metastatic_disease:regional_nodal_only" in rejections


def test_no_findings_gives_all_false():
    fields, rejections = evaluate_patient([])
    assert not fields["is_avpc"] and fields["n_avpc_criteria"] == 0
    assert all(fields[f"c{n}_met"] is False for n in range(1, 8))
    assert rejections == []


# --- Manual-review regressions (criteria v4) ----------------------------------------


@pytest.mark.parametrize("quote", [
    "Lytic metastasis in the L3 vertebral body.",
    "New 1.2 cm lytic lesion in the right iliac bone, consistent with metastasis.",
])
def test_c3_single_lytic_lesion_is_not_predominance(quote):
    lytic = row("bone_metastasis", quote, lytic_predominant=True)
    fields, rejections = evaluate_patient([CRPC, lytic])
    assert fields["c3_met"] is False
    assert "bone_metastasis:c3_single_lytic_lesion" in rejections


def test_c3_plural_lytic_contradicted_by_sclerotic_description_nearby():
    lytic = row("bone_metastasis", "Multiple lytic osseous metastases.", "2021-03-01",
                lytic_predominant=True)
    assert evaluate_patient([CRPC, lytic])[0]["c3_met"] is True
    sclerotic = row("bone_metastasis", "Diffuse sclerotic osseous metastases.", "2021-04-01",
                    lytic_predominant=False)
    fields, rejections = evaluate_patient([CRPC, lytic, sclerotic])
    assert fields["c3_met"] is False
    assert "bone_metastasis:c3_contradicted_by_blastic_description" in rejections
    # A sclerotic description years apart does not speak to this time point.
    old = {**sclerotic, "event_date": "2017-01-01", "source_note_date": "2017-01-01"}
    assert evaluate_patient([CRPC, lytic, old])[0]["c3_met"] is True


def test_c3_explicit_predominance_needs_explicit_blastic_predominance_to_contradict():
    lytic = row("bone_metastasis", "Predominantly lytic osseous metastases.", "2021-03-01",
                lytic_predominant=True)
    mixed = row("bone_metastasis", "Mixed lytic and sclerotic bone lesions.", "2021-04-01",
                lytic_predominant=False)
    assert evaluate_patient([CRPC, lytic, mixed])[0]["c3_met"] is True
    blastic = row("bone_metastasis", "Predominantly sclerotic bone metastases.", "2021-04-01",
                  lytic_predominant=False)
    assert evaluate_patient([CRPC, lytic, blastic])[0]["c3_met"] is False


@pytest.mark.parametrize("item,quote", [
    ("bulky_prostate_pelvic_mass", "CT abdomen and pelvis: 6.1 cm liver mass."),
    ("bulky_prostate_pelvic_mass", "Enlarged prostate; hepatic lesion measuring 6.1 cm."),
    ("bulky_lymphadenopathy", "Pelvic lymph nodes are stable; liver lesion measures 6.1 cm."),
])
def test_c4_measurement_must_belong_to_node_or_pelvic_mass(item, quote):
    finding = row(item, quote, size_cm=6.1)
    assert screen_finding(finding).startswith("c4_measurement_not_")
    grade = row("gleason_high_grade", "Gleason 4+5=9.", gleason_total=9)
    assert evaluate_patient([MET, CRPC, finding, grade])[0]["c4_met"] is False


def test_c4_pelvic_mass_still_counts_when_liver_also_mentioned():
    mass = row("bulky_prostate_pelvic_mass",
               "Prostate mass measuring 6.1 cm; small liver lesions.", size_cm=6.1)
    assert screen_finding(mass) is None


def test_c5_retold_diagnostic_psa_does_not_pair_with_later_bone_scan():
    # 2021 note restating the 2015 diagnostic PSA, with no stated date.
    psa = row("low_psa", "At diagnosis PSA was 6.2 ng/mL.", "2021-05-01", source="note_date",
              psa_ng_ml=6.2, psa_context="initial_presentation_pre_adt")
    bone = row("bone_metastasis", "Bone scan: innumerable osseous metastases.", "2021-05-10")
    fields, rejections = evaluate_patient([CRPC, psa, bone])
    assert fields["c5_met"] is False
    assert "low_psa:c5_presentation_psa_undated" in rejections


def test_c5_year_precision_never_pairs_and_pre_adt_psa_must_precede_adt():
    bone = row("bone_metastasis", "Bone scan: innumerable osseous metastases.", "2015-03-10")
    year_psa = row("low_psa", "PSA at diagnosis was 6.2.", "2015-01-01", "year",
                   psa_ng_ml=6.2, psa_context="initial_presentation_pre_adt")
    assert evaluate_patient([CRPC, year_psa, bone])[0]["c5_met"] is False

    psa = {**year_psa, "date_precision": "day", "event_date": "2015-03-01"}
    fields, _ = evaluate_patient([CRPC, psa, bone])
    assert fields["c5_met"] is True
    assert fields["c5_psa_context"] == "initial_presentation_pre_adt"
    assert fields["c5_gap_days"] == 9

    adt = row("adt_start", "Started leuprolide.", "2014-06-01", agent="leuprolide")
    fields, rejections = evaluate_patient([CRPC, psa, bone, adt])
    assert fields["c5_met"] is False
    assert "low_psa:c5_pre_adt_psa_after_adt_start" in rejections


def test_c5_castrate_psa_must_follow_adt_and_context_can_be_inferred_pre_adt():
    adt = row("adt_start", "Started leuprolide.", "2020-01-15", agent="leuprolide")
    bone = row("bone_metastasis", "Bone scan: innumerable osseous metastases.", "2019-12-20")
    early = row("low_psa", "At progression PSA was 3.0.", "2019-12-01", psa_ng_ml=3.0,
                psa_context="symptomatic_crpc_progression")
    fields, rejections = evaluate_patient([CRPC, adt, bone, early])
    assert fields["c5_met"] is False
    assert "low_psa:c5_castrate_psa_before_adt_start" in rejections
    unlabeled = row("low_psa", "PSA 3.0 ng/mL.", "2019-12-01", psa_ng_ml=3.0, psa_context=None)
    fields, _ = evaluate_patient([CRPC, adt, bone, unlabeled])
    assert fields["c5_met"] is True
    assert fields["c5_psa_context"] == "inferred_pre_adt"


@pytest.mark.parametrize("quote,high", [
    ("Bone scan: innumerable osseous metastases.", True),
    ("More than 20 osseous metastases.", True),
    ("25+ bone metastases on bone scan.", True),
    ("Osseous metastases; PSA greater than 20.", False),
    ("Bone metastasis with 25 mm lesions.", False),
    ("Innumerable pulmonary nodules and two bone metastases.", False),
    ("Extensive bone metastases.", False),
])
def test_c5_high_volume_counting(quote, high):
    psa = row("low_psa", "At progression PSA was 4.5 ng/mL.", psa_ng_ml=4.5,
              psa_context="symptomatic_crpc_progression")
    bone = row("bone_metastasis", quote)
    assert evaluate_patient([CRPC, psa, bone])[0]["c5_met"] is high


def test_c7_anchors_on_first_adt_not_a_later_precise_start():
    first = row("adt_start", "Started leuprolide in 2016.", "2016-01-01", "year",
                agent="leuprolide")
    later = row("adt_start", "Started ADT plus abiraterone.", "2020-01-01", "month",
                agent="abiraterone")
    crpc = _crpc("2020-05-20")
    fields, rejections = evaluate_patient([MET, first, later, crpc])
    assert fields["c7_met"] is False
    assert "adt_start:c7_first_adt_start_not_dated" in rejections
    # With no record of the 2016 start, an ARPI-only start is not an anchor.
    arpi = row("adt_start", "Started enzalutamide.", "2020-01-01", "month", agent="enzalutamide")
    assert evaluate_patient([MET, arpi, crpc])[0]["c7_met"] is False


def test_c7_crpc_before_anchor_means_anchor_is_not_first_adt():
    adt = _adt("2020-01-01")
    earlier_crpc = _crpc("2019-06-01")
    fields, rejections = evaluate_patient([MET, adt, earlier_crpc, _crpc("2020-05-20")])
    assert fields["c7_met"] is False
    assert "adt_start:c7_crpc_before_first_adt_start" in rejections


@pytest.mark.parametrize("quote", [
    "Progressed within 3 months of starting enzalutamide.",
    "Castration-resistant progression within 4 months of abiraterone.",
])
def test_c7_rapid_progression_on_second_line_agent_is_not_c7(quote):
    rapid = row("rapid_progression_on_adt", quote)
    fields, rejections = evaluate_patient([MET, rapid])
    assert fields["c7_met"] is False
    assert "rapid_progression_on_adt:c7_second_line_hormonal_therapy" in rejections


@pytest.mark.parametrize("quote", [
    "Lung nodules, stable, likely benign; no metastatic disease.",
    "Liver lesions, not consistent with metastases.",
    "Adrenal nodule consistent with adenoma.",
])
def test_visceral_requires_an_asserted_malignancy_word(quote):
    assert screen_finding(row("visceral_metastasis", quote, organ="liver")).startswith(
        "no_metastatic_assertion")


def test_negated_visceral_does_not_make_patient_metastatic():
    fields, _ = evaluate_patient([CRPC, row(
        "visceral_metastasis", "Liver lesions, not consistent with metastases.", organ="liver")])
    assert fields["is_metastatic"] is not True


def test_gleason_needs_a_gleason_context():
    assert not gleason_in_quote(8, "PSA = 8")
    assert gleason_in_quote(8, "Gleason 4+4=8")
    assert gleason_in_quote(9, "GS 9")
