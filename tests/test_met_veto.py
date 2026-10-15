"""Tests for the metastatic-disease quote gate (tasks/longitudinal_subtype/met_veto.py).

The mCRPC gate's metastatic screen defers to this gate, so these are the
executable form of its precision contract. Two boundaries carry most of the
weight:

  - M1 vs. N1: regional pelvic nodal disease alone is NOT metastatic, despite
    pathology reports describing it with the word "metastatic".
  - destination vs. primary: "metastatic prostate cancer to the lung" names a
    lung LESION and qualifies, while "known metastatic lung primary" names a
    lung CANCER and does not.
"""

import pytest

from tasks.longitudinal_subtype.met_veto import screen_quote


# --------------------------------------------------------------------------
# Deterministic quote gate -- accepts
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "quote",
    [
        "IMPRESSION: multiple osseous metastases consistent with metastatic prostate cancer.",
        "ASSESSMENT: mCRPC with liver metastases.",
        "Problem list: metastatic prostate cancer to bone.",
        "Bone scan demonstrates widespread osseous metastases.",
        "He has M1b prostate cancer.",
        "s/p radium-223 for osseous metastases.",
        "IMPRESSION: innumerable sclerotic osseous lesions consistent with"
        " metastatic disease.",
        "Pathology: metastatic adenocarcinoma in retroperitoneal lymph node,"
        " consistent with prostate primary.",
    ],
)
def test_veto_accepts_asserted_metastatic_disease(quote):
    ok, reason = screen_quote(quote)
    assert ok, reason


def test_veto_keeps_stable_metastatic_disease():
    """Copy-forward guard.

    "stable"/"unchanged" are status words in an oncology assessment, not
    negations -- "stable metastatic disease on abiraterone" is one of the most
    common ways a positive is written, and vetoing it would lose a large share
    of the cohort.
    """
    ok, reason = screen_quote("Stable metastatic disease on abiraterone.")
    assert ok, reason


def test_veto_keeps_metastasis_to_a_non_prostate_site():
    """A named organ near the term is usually a DESTINATION, not a primary."""
    ok, reason = screen_quote("Metastatic prostate cancer to the lung.")
    assert ok, reason


# --------------------------------------------------------------------------
# Deterministic quote gate -- rejects
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "quote",
    [
        "No evidence of metastatic disease.",
        "Bone scan negative for osseous metastases.",
        "Lesion suspicious for osseous metastasis.",
        "Findings concerning for metastatic disease.",
        "Cannot exclude metastatic disease.",
        "Mild degenerative changes; no evidence of osseous metastatic disease.",
        "Monitoring for metastatic progression.",
        "At risk for metastatic disease.",
        "Restaging to rule out metastasis.",
    ],
)
def test_veto_rejects_negated_and_hedged_quotes(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("negated_or_hedged:")


@pytest.mark.parametrize(
    "quote",
    [
        # A site descriptor with nothing asserting spread.
        "Sclerotic lesions in the pelvis.",
        "No new osseous lesions.",
    ],
)
def test_veto_rejects_site_descriptors_without_an_assertion(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "no_metastatic_assertion"


def test_veto_rejects_uptake_without_a_site_term():
    """"Increased uptake in the right rib" carries no met term at all.

    It is retrieved by the "bone scan" trigger, not by a metastasis term, so it
    fails at the first gate rather than the assertion gate.
    """
    ok, reason = screen_quote("Increased uptake in the right rib.")
    assert not ok
    assert reason == "no_met_term"


@pytest.mark.parametrize(
    "quote",
    [
        "Sclerotic focus likely degenerative.",
        "PSA 12.4; continue ADT.",
        "",
        None,
    ],
)
def test_veto_rejects_quotes_without_a_met_term(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "no_met_term"


@pytest.mark.parametrize(
    "quote",
    [
        # N1, not M1 -- the sharpest boundary in this task. Prostatectomy
        # pathology describes regional nodal involvement with the word
        # "metastatic", and treating it as M1 would mislabel a large number of
        # locally advanced, non-metastatic patients.
        "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes.",
        "Metastatic carcinoma involving obturator lymph nodes.",
        "Metastatic disease in the internal iliac nodes.",
    ],
)
def test_veto_rejects_regional_nodal_only_disease(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason == "regional_nodal_only"


def test_regional_nodes_alongside_a_distant_site_still_qualify():
    """The N1 check is a property of the quote's whole site set."""
    ok, reason = screen_quote(
        "Metastatic disease involving pelvic and retroperitoneal lymph nodes."
    )
    assert ok, reason


def test_regional_nodes_with_bone_disease_still_qualify():
    ok, reason = screen_quote(
        "Metastatic prostate cancer involving pelvic nodes and the lumbar spine."
    )
    assert ok, reason


@pytest.mark.parametrize(
    "quote",
    [
        "Known metastatic lung primary with osseous lesions.",
        "History of metastatic colon cancer.",
        "Metastatic melanoma, primary resected in 2015.",
    ],
)
def test_veto_rejects_a_non_prostate_primary(quote):
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("non_prostate_primary:")


@pytest.mark.parametrize(
    "quote",
    [
        "Inclusion criteria: patients with metastatic castration-resistant"
        " prostate cancer are eligible.",
        "This is a Phase III study of patients with metastatic prostate cancer.",
        "Cohort B: metastatic hormone-sensitive prostate cancer.",
    ],
)
def test_veto_rejects_clinical_trial_boilerplate(quote):
    """Trial text names metastatic disease because the trial targets it."""
    ok, reason = screen_quote(quote)
    assert not ok
    assert reason.startswith("boilerplate:")
