import pytest

from tasks.arlpc.ar_gate import (
    ar_target_loss,
    screen_ar_ihc,
    screen_genomic,
    screen_statement,
)


@pytest.mark.parametrize("quote, result", [
    ("Tumor cells are negative for AR.", "negative"),
    ("AR: negative.", "negative"),
    ("Androgen receptor shows weak staining.", "reduced"),
    ("AR is focally positive (10%).", "reduced"),
    ("AR weak but diffuse.", "reduced"),
    ("AR shows heterogeneous staining in the metastasis.", "reduced"),
    ("AR expression is reduced compared with the prior prostatectomy.", "reduced"),
    ("Loss of androgen receptor expression in the liver lesion.", "negative"),
    ("Positive for PSA, negative for AR.", "negative"),
])
def test_ar_low_qualifies(quote, result):
    assert screen_ar_ihc(quote, result) == (True, None, True)


@pytest.mark.parametrize("quote", [
    "AR positive (strong, diffuse).",
    "Androgen receptor shows moderate staining.",
    "AR is weak to moderate in intensity.",
])
def test_ar_positive_is_audit_only(quote):
    assert screen_ar_ihc(quote, "positive") == (True, None, False)
    ok, reason, _ = screen_ar_ihc(quote, "reduced")
    assert not ok and reason == "ar_result_not_in_quote"


def test_panel_binds_per_marker():
    quote = "Negative for AR and positive for NKX3.1."
    assert screen_ar_ihc(quote, "negative")[0]
    assert ar_target_loss(quote) == ([], "no_low_target_result")


@pytest.mark.parametrize("quote, reason", [
    ("AR stain is pending.", "ihc_not_resulted:pending"),
    ("AR and NKX3.1 negative, favor urothelial carcinoma.", "non_prostate_origin:urothelial"),
    ("AR negative, rather than prostatic origin.", "non_prostate_origin:against_prostate"),
    ("AR-V7 negative on ctDNA.", "no_ar_marker"),
])
def test_ar_ihc_rejections(quote, reason):
    ok, got, _ = screen_ar_ihc(quote, "negative")
    assert not ok and got == reason


def test_model_origin_vetoes_and_missing_result():
    assert screen_ar_ihc("AR negative.", "negative", "non_prostate")[1] == "non_prostate_origin:model"
    assert screen_ar_ihc("AR negative.", None)[1] == "missing_ar_result"


@pytest.mark.parametrize("quote", [
    "No loss of AR expression.",
    "AR expression is not reduced.",
    "AR shows no loss of expression.",
    "AR staining is not weak.",
    "AR is not negative.",
    "Loss of AR expression is not identified.",
])
@pytest.mark.parametrize("result", ["negative", "reduced"])
def test_negated_low_result_cannot_establish_ar_low(quote, result):
    assert screen_ar_ihc(quote, result)[0] is False


@pytest.mark.parametrize("quote", [
    "AR strong, NKX3.1 negative.",
    "NKX3.1 negative, AR strong.",
    "AR strong. NKX3.1 negative.",
    "NKX3.1 negative; AR strong.",
    "AR strong and NKX3.1 negative.",
    "NKX3.1 negative and AR strong.",
])
def test_neighbor_negative_cannot_override_strong_ar(quote):
    assert screen_ar_ihc(quote, "positive") == (True, None, False)
    assert screen_ar_ihc(quote, "negative")[0] is False
    assert ar_target_loss(quote, marker="nkx3_1") == (["nkx3_1"], None)


@pytest.mark.parametrize("quote", [
    "AR and NKX3.1 negative.",
    "Negative for NKX3.1 and AR.",
    "AR, NKX3.1 and PSAP are negative.",
])
def test_coordinated_markers_keep_their_shared_result(quote):
    assert screen_ar_ihc(quote, "negative") == (True, None, True)
    assert "nkx3_1" in ar_target_loss(quote)[0]


@pytest.mark.parametrize("quote", [
    "AR negative, consistent with lung metastasis from prostate cancer.",
    "AR negative, consistent with pulmonary metastases of prostatic origin.",
    "AR negative, consistent with metastatic prostate carcinoma in lung.",
])
def test_prostate_metastatic_destination_does_not_veto(quote):
    assert screen_ar_ihc(quote, "negative", "prostate") == (True, None, True)
    assert screen_ar_ihc(quote, "negative")[0] is True
    assert screen_ar_ihc(quote, "negative", "non_prostate")[0] is False


@pytest.mark.parametrize("quote", [
    "History of prostate cancer. AR negative, consistent with lung carcinoma.",
    "AR negative, favors urothelial carcinoma rather than prostate cancer.",
    "AR negative, consistent with lung metastasis from urothelial carcinoma.",
])
def test_prostate_history_does_not_excuse_other_primary(quote):
    assert screen_ar_ihc(quote, "negative", "prostate")[0] is False


def test_ar_target_loss_and_serum_psa():
    assert ar_target_loss("Tumor cells are negative for NKX3.1 and PSAP.") == (["nkx3_1", "psap"], None)
    assert ar_target_loss("PSA immunostain is weak.") == (["psa"], None)
    assert ar_target_loss("PSA negative.", note_type="Clinician") == ([], "psa_not_tissue")
    assert ar_target_loss("Serum PSA negative trend, PSA stain negative.")[1] == "psa_serum"
    assert ar_target_loss("PSMA negative on PET, stain not done.")[1] == "psma_not_ihc"
    assert ar_target_loss("NKX3.1 negative.", marker="psap")[1] == "marker_not_low_in_quote"


@pytest.mark.parametrize("quote, evidence_type, ok", [
    ("AR-negative prostate cancer.", "ar_statement", True),
    ("Double-negative (AR-/NE-) mCRPC.", "ar_statement", True),
    ("DNPC, AR-null and NE-negative.", "ar_statement", True),
    ("Not AR-negative.", "ar_statement", False),
    ("Possible AR-indifferent disease.", "ar_statement", False),
    ("Androgen-independent prostate cancer.", "ar_statement", False),
    ("Androgen-independent prostate cancer.", "androgen_independent_statement", True),
    ("Hormone-refractory prostate cancer since 2015.", "androgen_independent_statement", True),
    ("Concern for hormone-refractory disease.", "androgen_independent_statement", False),
    ("Inclusion criteria: AR-null mCRPC.", "ar_statement", False),
])
def test_statements(quote, evidence_type, ok):
    assert screen_statement(quote, evidence_type)[0] is ok


@pytest.mark.parametrize("quote, ok, reason", [
    ("Foundation One: AR homozygous deletion.", True, None),
    ("ctDNA shows AR copy number loss.", True, None),
    ("Germline testing: AR deletion.", False, "genomic_ar_driven:germline"),
    ("AR amplification and AR deletion of exon 3 on NGS.", False, "genomic_ar_driven:amplification"),
    ("AR loss.", False, "no_genomic_context"),
])
def test_genomic(quote, ok, reason):
    assert screen_genomic(quote) == (ok, reason)
