import pytest

from tasks.nepc.ne_gate import positive_ihc_markers, screen_asserted, screen_ne_quote, NE_TERM


@pytest.mark.parametrize("quote,expected", [
    ("PSA positive. Synaptophysin staining is entirely negative.", []),
    ("PSA positive, synaptophysin is completely negative.", []),
    ("Synaptophysin negative. Chromogranin strongly positive.", ["chromogranin"]),
    ("Synaptophysin negative, chromogranin strongly positive.", ["chromogranin"]),
    ("Synaptophysin negative and chromogranin strongly positive.", ["chromogranin"]),
    ("Synaptophysin and chromogranin A are positive.", ["chromogranin a", "synaptophysin"]),
    ("Positive for synaptophysin, chromogranin A and CD56.",
     ["cd56", "chromogranin a", "synaptophysin"]),
])
def test_panel_polarity_stays_with_its_marker_clause(quote, expected):
    assert positive_ihc_markers(quote)[0] == expected


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


@pytest.mark.parametrize("quote,expected", [
    ("Focal weak synaptophysin staining", True),
    ("Diffuse synaptophysin staining", True),
    ("No focal weak synaptophysin staining", False),
    ("Focal weak synaptophysin staining is not identified", False),
    ("Focal weak synaptophysin staining is nonspecific", False),
    ("Focal weak synaptophysin staining pending", False),
])
def test_explicit_stain_intensity_is_a_result_only_when_asserted(quote, expected):
    assert screen_ne_quote(quote, "positive_ne_ihc")[0] is expected


@pytest.mark.parametrize("quote,evidence_type", [
    ("Discussed with patient the possibility of NEPC.", "nepc_diagnosis"),
    ("NE markers were negative; small cell carcinoma unlikely.", "small_cell_histology"),
    ("Small cell carcinoma less likely.", "small_cell_histology"),
    ("Small cell carcinoma, unlikely.", "small_cell_histology"),
    ("Small cell carcinoma - doubtful.", "small_cell_histology"),
    ("Question of small cell carcinoma.", "small_cell_histology"),
    ("?NEPC", "nepc_diagnosis"),
    ("NEPC?", "nepc_diagnosis"),
    ("Plan repeat biopsy to look for small cell carcinoma.", "small_cell_histology"),
    ("Hypothesis of treatment-emergent NEPC.", "histologic_transformation"),
    ("Tumor shows neuroendocrine features? Awaiting stains", "ne_differentiation"),
])
def test_v6_hedges_and_questions_are_rejected(quote, evidence_type):
    ok, reason = screen_ne_quote(quote, evidence_type)
    assert not ok and reason.startswith("negated_or_hedged:")


def test_question_in_another_sentence_does_not_hedge():
    quote = "Prostate biopsy: small cell carcinoma. Is this new? Yes."
    assert screen_ne_quote(quote, "small_cell_histology") == (True, None)


def test_screen_any_asserted_needs_one_asserted_occurrence():
    from tasks.nepc.ne_gate import screen_any_asserted
    assert screen_any_asserted("No NEPC before. Now NEPC confirmed.", NE_TERM) == (True, None)
    assert screen_any_asserted("No NEPC.", NE_TERM)[0] is False
    assert screen_any_asserted("Adenocarcinoma.", NE_TERM) == (False, "no_term")
