import pytest

from tasks.nepc.ne_gate import positive_ihc_markers, screen_asserted, screen_ne_quote, NE_TERM


@pytest.mark.parametrize(
    "quote,evidence_type",
    [
        ("Small cell carcinoma of the prostate, extensive.", "nepc_diagnosis"),
        ("No perineural invasion. Small cell carcinoma, extensive.", "nepc_diagnosis"),
        ("ASSESSMENT: metastatic neuroendocrine prostate cancer, s/p carboplatin/etoposide",
         "nepc_diagnosis"),
        ("Liver, biopsy: high-grade neuroendocrine carcinoma, consistent with prostatic origin",
         "nepc_diagnosis"),
        ("biopsy showed transformation to small cell carcinoma", "histologic_transformation"),
        ("adenocarcinoma with focal neuroendocrine differentiation", "ne_differentiation"),
        ("Tumor cells are positive for synaptophysin and negative for chromogranin.",
         "positive_ne_ihc"),
        ("negative for CD56 but positive for synaptophysin", "positive_ne_ihc"),
        ("INSM1 (+), chromogranin (-)", "positive_ne_ihc"),
        ("Small cell carcinoma of the prostate", "small_cell_histology"),
    ],
)
def test_asserted_ne_quotes_pass(quote, evidence_type):
    assert screen_ne_quote(quote, evidence_type) == (True, None)


@pytest.mark.parametrize(
    "quote,evidence_type,reason",
    [
        ("No small cell component is identified.", "nepc_diagnosis", "negated_or_hedged:no"),
        ("small cell carcinoma is not identified", "nepc_diagnosis",
         "negated_or_hedged:post_negation"),
        ("possible small cell carcinoma", "nepc_diagnosis", "negated_or_hedged:possible"),
        ("concern for neuroendocrine transformation", "histologic_transformation",
         "negated_or_hedged:concern"),
        ("history of small cell lung cancer", "nepc_diagnosis", "non_prostate_primary:lung"),
        ("non-small cell lung carcinoma", "nepc_diagnosis", "non_small_cell"),
        ("Merkel cell carcinoma, a neuroendocrine carcinoma of skin", "nepc_diagnosis",
         "non_prostate_primary:merkel"),
        ("Eligibility: patients with small cell carcinoma of the prostate", "nepc_diagnosis",
         "boilerplate:eligibility"),
        ("Synaptophysin, chromogranin and CD56 are negative.", "positive_ne_ihc", None),
        ("negative for synaptophysin, chromogranin and CD56", "positive_ne_ihc", "negative_ihc"),
        ("synaptophysin stain pending", "positive_ne_ihc", None),
        ("Serum chromogranin A elevated at 450 ng/mL", "positive_ne_ihc", None),
        ("adenocarcinoma, acinar type", "nepc_diagnosis", "no_ne_diagnosis_term"),
        ("transformation noted", "histologic_transformation", "no_ne_term"),
        ("small cell carcinoma", "made_up_type", "invalid_evidence_type:made_up_type"),
    ],
)
def test_negated_hedged_or_foreign_ne_quotes_fail(quote, evidence_type, reason):
    ok, got = screen_ne_quote(quote, evidence_type)
    assert ok is False
    if reason is not None:
        assert got == reason


def test_positive_markers_are_read_per_marker():
    markers, reason = positive_ihc_markers(
        "Tumor cells are positive for synaptophysin and chromogranin A, negative for CD56."
    )
    assert markers == ["chromogranin a", "synaptophysin"]
    assert reason is None


def test_screen_asserted_is_sentence_bounded():
    assert screen_asserted("No PNI. Neuroendocrine carcinoma present.", NE_TERM) == (True, None)
    assert screen_asserted("No neuroendocrine carcinoma.", NE_TERM)[0] is False
    assert screen_asserted("acinar adenocarcinoma", NE_TERM) == (False, "no_term")


@pytest.mark.parametrize(
    "quote",
    [
        "No positive synaptophysin staining.",
        "No synaptophysin positivity is identified.",
        "There is no convincing synaptophysin expression.",
        "Synaptophysin expression is not present.",
        "Synaptophysin immunoreactivity was not observed.",
        "No synaptophysin and chromogranin positivity is identified.",
        "No positive staining for synaptophysin and chromogranin.",
    ],
)
def test_negated_positive_ihc_language_is_not_a_positive_result(quote):
    assert positive_ihc_markers(quote) == ([], "negative_ihc")
    assert screen_ne_quote(quote, "positive_ne_ihc")[0] is False


@pytest.mark.parametrize(
    "quote,expected",
    [
        ("No synaptophysin positivity, chromogranin positive.", ["chromogranin"]),
        ("No synaptophysin positivity and chromogranin positive.", ["chromogranin"]),
        ("No synaptophysin staining but chromogranin is positive.", ["chromogranin"]),
        ("Synaptophysin positive, no chromogranin positivity.", ["synaptophysin"]),
        ("No perineural invasion. Synaptophysin is positive.", ["synaptophysin"]),
        ("Negative for CD56 but positive for synaptophysin.", ["synaptophysin"]),
    ],
)
def test_negated_ihc_result_does_not_negate_other_positive_markers(quote, expected):
    assert positive_ihc_markers(quote) == (expected, None)


@pytest.mark.parametrize(
    "quote",
    [
        "Small cell carcinoma cannot be excluded.",
        "Small cell carcinoma is suspected.",
        "Small cell carcinoma is a possibility.",
        "Small cell carcinoma, favored.",
        "Small cell carcinoma may be present.",
    ],
)
def test_postposed_diagnostic_uncertainty_is_rejected(quote):
    assert screen_ne_quote(quote, "nepc_diagnosis") == (
        False, "negated_or_hedged:post_hedge"
    )
    assert screen_ne_quote(quote, "small_cell_histology")[0] is False
    assert screen_asserted(quote, NE_TERM)[0] is False


def test_postposed_hedge_check_stops_at_next_sentence():
    assert screen_ne_quote(
        "Small cell carcinoma. Lymphovascular invasion is suspected.", "nepc_diagnosis"
    ) == (True, None)


@pytest.mark.parametrize(
    "quote",
    [
        "Adenocarcinoma with squamous transdifferentiation.",
        "Lineage plasticity is present in the prostate tumor.",
    ],
)
def test_transformation_requires_a_neuroendocrine_target(quote):
    assert screen_ne_quote(quote, "histologic_transformation") == (False, "no_ne_term")
    assert screen_ne_quote(quote, "ne_differentiation") == (False, "no_ne_term")


@pytest.mark.parametrize(
    "quote",
    [
        "Prostate adenocarcinoma with transdifferentiation to small cell carcinoma.",
        "Lineage plasticity to a neuroendocrine phenotype has occurred.",
        "Treatment-emergent NEPC.",
    ],
)
def test_transformation_with_explicit_neuroendocrine_target_passes(quote):
    assert screen_ne_quote(quote, "histologic_transformation") == (True, None)


def test_neuroendocrine_target_does_not_make_negated_transformation_asserted():
    assert screen_ne_quote(
        "No transdifferentiation. Neuroendocrine carcinoma is present.",
        "histologic_transformation",
    ) == (False, "negated_or_hedged:no")


def test_trial_population_requirements_are_not_patient_findings():
    diagnosis = "Participants must have small cell carcinoma of the prostate."
    ihc = "Participants must have positive synaptophysin staining."
    assert screen_ne_quote(diagnosis, "nepc_diagnosis")[0] is False
    assert screen_asserted(diagnosis, NE_TERM)[0] is False
    assert screen_ne_quote(ihc, "positive_ne_ihc")[0] is False
