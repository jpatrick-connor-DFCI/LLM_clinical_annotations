import pytest

from tasks.avpc_adjacent.criteria import evaluate_patient, screen_finding


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


def visceral(quote, date="2021-01-15", **fields):
    return row("visceral_metastasis", quote, date, exclusively_visceral=False, **fields)


def bone(quote, **fields):
    return row("bone_metastasis", quote, **fields)


def av(quote):
    return row("aggressive_variant_statement", quote, term="aggressive variant")


# --- A1: extended visceral ----------------------------------------------------------


@pytest.mark.parametrize("quote", [
    "Multiple liver metastases from prostate adenocarcinoma.",
    "New dural-based metastasis along the left convexity.",
    "MRI shows leptomeningeal carcinomatosis consistent with metastatic prostate cancer.",
    "Biopsy of subcutaneous nodule: metastatic prostatic adenocarcinoma; subcutaneous metastasis.",
    "Soft tissue metastasis in the left gluteal muscle.",
    "Metastatic prostate cancer with bladder metastases.",
])
def test_extended_visceral_sites_accepted(quote):
    assert screen_finding(visceral(quote)) is None


def test_no_state_gate_visceral_alone_is_adjacent():
    fields, _ = evaluate_patient([visceral("Hepatic metastases, biopsy-proven prostate adenocarcinoma.",
                                           "2019-05-01")])
    assert fields["a1_visceral_met"] is True
    assert fields["is_castration_resistant"] is False
    assert fields["is_avpc_adjacent"] is True
    assert fields["criteria"] == ["A1"]
    assert fields["avpc_adjacent_date"] == "2019-05-01"


@pytest.mark.parametrize("quote", [
    "Metastatic prostate cancer to retroperitoneal lymph nodes.",
    "Mesenteric lymph node metastases.",
    "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes.",
])
def test_lymph_nodes_never_visceral(quote):
    assert screen_finding(visceral(quote)) is not None


def test_bone_lesion_soft_tissue_extension_is_not_visceral():
    quote = "L3 lytic metastasis with associated soft tissue mass extending into the epidural space."
    assert screen_finding(visceral(quote)) == "bone_extension_not_visceral"
    quote = "Sacral metastasis with soft tissue component; soft tissue mass arising from the sacrum."
    assert screen_finding(visceral(quote)) == "bone_extension_not_visceral"


@pytest.mark.parametrize("quote", [
    "Sacral metastasis extending into muscle; muscle involvement.",
    "Calvarial metastasis with direct dural involvement.",
    "Muscle involvement extending from a sacral metastasis.",
    "T12 metastasis with associated intramuscular mass.",
])
def test_named_tissue_extension_does_not_set_adjacent(quote):
    fields, rejections = evaluate_patient([visceral(quote)])
    assert fields["is_avpc_adjacent"] is False
    assert "visceral_metastasis:bone_extension_not_visceral" in rejections


@pytest.mark.parametrize("quote", [
    "Bone metastasis with muscle metastasis.",
    "Muscle metastasis. Sacral metastasis with soft tissue extension.",
    "Calvarial bone metastasis. Separate dural metastasis.",
    "Soft tissue metastasis without extension from bone.",
    "L3 metastasis with associated soft tissue mass; hepatic metastases.",
])
def test_separate_metastasis_survives_bone_extension_screen(quote):
    fields, _ = evaluate_patient([visceral(quote)])
    assert fields["a1_visceral_met"] is True


def test_t4_invasion_is_not_a_bladder_metastasis():
    assert screen_finding(visceral("Prostate tumor invades the bladder neck.")) is not None
    quote = "Metastatic prostate cancer directly invading the bladder; bladder metastasis by extension."
    assert screen_finding(visceral(quote)) == "local_invasion_not_metastasis"


@pytest.mark.parametrize("quote", [
    "Bladder metastasis. No local invasion.",
    "Bladder metastasis without local invasion of the bladder.",
    "Rectal metastasis; local invasion is absent.",
    "Bladder metastasis. Prostate tumor invades the seminal vesicles.",
])
def test_negated_or_unrelated_invasion_preserves_metastasis(quote):
    fields, _ = evaluate_patient([visceral(quote)])
    assert fields["is_avpc_adjacent"] is True
    assert fields["a1_visceral_met"] is True


def test_negated_visceral_rejected():
    assert screen_finding(visceral("No liver metastases. Metastatic disease to bone.")).startswith(
        "negated_or_hedged")


# --- A2: any lytic ------------------------------------------------------------------


def test_mixed_lytic_blastic_counts_but_strict_c3_does_not():
    b = bone("Mixed lytic and sclerotic osseous metastases in the spine.",
             lytic_component=True, lytic_predominant=False)
    fields, _ = evaluate_patient([b])
    assert fields["a2_lytic_met"] is True
    assert fields["c3_met"] is False
    assert fields["criteria"] == ["A2"]


def test_blastic_only_and_negated_lytic_rejected():
    blastic = bone("Sclerotic osseous metastases.", lytic_component=False, lytic_predominant=False)
    negated = bone("Blastic bone metastases, no lytic lesions.", lytic_component=True)
    fields, rejections = evaluate_patient([blastic, negated])
    assert fields["a2_lytic_met"] is False
    # The metastatic gate or the A2 lytic screen rejects the negated quote.
    assert any(r.startswith("bone_metastasis:") and "negated_or_hedged" in r for r in rejections)


# --- A3: modified C5 ----------------------------------------------------------------

PSA = row("low_psa", "At symptomatic progression PSA was 3.1 ng/mL.", "2021-02-01",
          psa_ng_ml=3.1, psa_context="symptomatic_crpc_progression")


@pytest.mark.parametrize("quote", [
    "Diffuse osseous metastases throughout the axial skeleton.",
    "Extensive bone metastases.",
    "Widespread osseous metastatic disease.",
    "Multiple bone metastases.",
    "High-volume bone metastases.",
    "Metastatic prostate cancer with heavy bone burden.",
    "Extensive mixed lytic and blastic bone metastases.",
    "Multiple new bone metastases.",
])
def test_a3_extent_wording_with_low_psa(quote):
    fields, _ = evaluate_patient([PSA, bone(quote, high_volume_descriptor="extensive")])
    assert fields["a3_c5mod_met"] is True
    assert fields["c5_met"] is False
    assert fields["a3_psa_quote"] == PSA["quote"]


def test_a3_needs_low_psa_and_negated_extent_fails():
    fields, _ = evaluate_patient([bone("Extensive bone metastases.")])
    assert fields["a3_c5mod_met"] is False
    fields, _ = evaluate_patient([PSA, bone("Osseous metastases, not extensive bone disease.")])
    assert fields["a3_c5mod_met"] is False


def test_a3_and_c5_do_not_borrow_liver_lesion_count():
    fields, _ = evaluate_patient([PSA, bone("2 bone metastases and 25 liver metastases.")])
    assert fields["a3_c5mod_met"] is False
    assert fields["c5_met"] is False


@pytest.mark.parametrize("quote", [
    "Single sclerotic bone metastasis with low bone burden.",
    "Small bone burden from one sclerotic metastasis.",
    "Single bone metastasis. Multiple nodal metastases.",
    "Single bone metastasis. Extensive lymph node metastases.",
    "Multiple nodal metastases and a single bone metastasis.",
    "Multiple nodal and bone lesions with only one bone metastasis.",
])
def test_a3_requires_extent_of_bone_disease(quote):
    fields, _ = evaluate_patient([
        PSA, bone(quote, count=1, high_volume_descriptor="extensive"),
    ])
    assert fields["is_avpc_adjacent"] is False
    assert fields["a3_c5mod_met"] is False


def test_a3_strict_volume_still_counts_and_c5_follows():
    fields, _ = evaluate_patient([PSA, bone("Innumerable osseous metastases.")])
    assert fields["a3_c5mod_met"] and fields["c5_met"]


# --- A4: explicit statement ---------------------------------------------------------


@pytest.mark.parametrize("quote", [
    "Clinical picture consistent with aggressive variant prostate cancer.",
    "He has AVPC and will start carboplatin.",
    "Anaplastic prostate cancer.",
    "Variant CRPC with visceral spread.",
    "He has AVPC. Start carboplatin?",
])
def test_av_statement_accepted(quote):
    assert screen_finding(av(quote)) is None


@pytest.mark.parametrize("quote", [
    "Concern for aggressive variant prostate cancer.",
    "?AVPC, will rebiopsy.",
    "Rule out aggressive variant.",
    "No features to suggest AVPC.",
    "History of anaplastic thyroid carcinoma.",
    "Anaplastic large cell lymphoma.",
    "Eligibility: patients with aggressive variant prostate cancer per protocol.",
])
def test_av_statement_rejected(quote):
    assert screen_finding(av(quote)) is not None


@pytest.mark.parametrize("quote", [
    "AVPC?",
    "Aggressive variant prostate cancer?",
    "Anaplastic prostate carcinoma?",
    "Variant CRPC?",
    "AVPC (?)",
])
def test_questioned_diagnosis_does_not_set_a4(quote):
    fields, rejections = evaluate_patient([av(quote)])
    assert fields["is_avpc_adjacent"] is False
    assert fields["a4_statement_met"] is False
    assert "aggressive_variant_statement:negated_or_hedged:question" in rejections


# --- Ungated strict criteria --------------------------------------------------------


def test_strict_c4_without_crpc_is_adjacent_not_avpc():
    node = row("bulky_lymphadenopathy", "Pelvic lymph node measuring 55 mm.", size_cm=5.5)
    met = row("metastatic_disease", "Metastatic prostate cancer to bone.", "2020-01-10")
    fields, _ = evaluate_patient([met, node])
    assert fields["c4_met"] is True
    assert fields["is_avpc_strict"] is False
    assert fields["is_avpc_adjacent"] is True
    assert fields["criteria"] == ["C4"]


def test_no_findings_is_not_adjacent():
    fields, rejections = evaluate_patient([])
    assert fields["is_avpc_adjacent"] is False and fields["criteria"] == [] and rejections == []


@pytest.mark.parametrize("quote", [
    "Adrenal nodule consistent with adenoma.",
    "Liver lesions, not consistent with metastases.",
    "Lung nodules, stable, likely benign; no metastatic disease.",
])
def test_benign_or_negated_visceral_is_not_a1(quote):
    assert screen_finding(visceral(quote)).startswith("no_metastatic_assertion")
    fields, _ = evaluate_patient([visceral(quote)])
    assert not fields["is_avpc_adjacent"]
