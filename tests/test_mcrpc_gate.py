import pytest

from tasks.longitudinal_subtype.mcrpc_gate import (
    CASTRATION_RESISTANT,
    METASTATIC,
    STATE_SCREENS,
    screen_castration_resistant_quote,
    screen_metastatic_quote,
)


@pytest.mark.parametrize(
    "quote",
    [
        "Assessment: mCRPC, on enzalutamide.",
        "Metastatic castration-resistant prostate cancer with lytic bone metastases.",
        "Bone scan demonstrates at least 24 osseous metastases.",
        "mHSPC, started ADT plus abiraterone.",
        "Stage IV prostate cancer.",
    ],
)
def test_metastatic_screen_accepts_affirmative_distant_disease(quote):
    assert screen_metastatic_quote(quote) == (True, None)


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        # The acronym expands to "non-metastatic ...", which the met gate negates.
        ("nmCRPC on apalutamide.", "negated_or_hedged:non"),
        ("No evidence of metastatic disease.", "negated_or_hedged:no"),
        # M1 vs N1: regional pelvic nodes alone are not metastatic disease.
        ("Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes.", "regional_nodal_only"),
        ("Progression to mCRPC in 2020.", "negated_or_hedged:progression to"),
        ("Eligible for the phase 3 trial in mCRPC.", "boilerplate:eligible"),
        ("CRPC with rising PSA.", "no_met_term"),
    ],
)
def test_metastatic_screen_rejects_non_assertions(quote, reason):
    assert screen_metastatic_quote(quote) == (False, reason)


@pytest.mark.parametrize(
    "quote",
    [
        "Assessment: mCRPC, on enzalutamide.",
        # Castration-resistant even though not metastatic.
        "nmCRPC on apalutamide.",
        "Non-metastatic castration-resistant prostate cancer.",
        "PSA progression on leuprolide.",
        "Rising PSA despite ADT, consistent with castration resistant disease.",
        # Affirmative here, although the met gate reads "progression to" as a hedge.
        "Progression to castration resistance in 2019.",
        "He developed castration-resistant disease in 2019.",
        "PSA increased from 2.1 to 7.4 on Lupron.",
        "Disease refractory to androgen deprivation therapy.",
        "Resistant to castration.",
        "Failed ADT; now on docetaxel.",
        "Progressed on abiraterone and enzalutamide.",
        "PSA progression despite castrate levels of testosterone.",
    ],
)
def test_castration_resistance_screen_accepts_resistance_on_castration(quote):
    assert screen_castration_resistant_quote(quote) == (True, None)


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        # The rise came before castration, not on it.
        ("PSA rose and he was started on ADT.", "no_castration_resistance_term"),
        ("mHSPC with PSA rising on ADT.", "castration_sensitive:mhspc"),
        ("Not yet castration resistant.", "negated_or_hedged:not"),
        ("He remains hormone sensitive.", "no_castration_resistance_term"),
        ("PSA rose during ADT holiday.", "off_castration:holiday"),
        (
            "Eligibility: patients with mCRPC who progressed on abiraterone.",
            "boilerplate:eligibility",
        ),
        ("Risk of developing castration resistance discussed.", "negated_or_hedged:risk of"),
        # A rise on therapy must be a PSA rise.
        ("Increased bone pain on ADT.", "no_castration_resistance_term"),
        ("At progression PSA was 7.2 ng/mL.", "no_castration_resistance_term"),
        ("Biochemical recurrence after RP started on ADT.", "no_castration_resistance_term"),
        # Treatment alone never establishes the state.
        ("Started enzalutamide.", "no_castration_resistance_term"),
        # Bicalutamide monotherapy does not castrate.
        ("PSA progression on bicalutamide monotherapy.", "no_castration_resistance_term"),
    ],
)
def test_castration_resistance_screen_rejects_non_assertions(quote, reason):
    assert screen_castration_resistant_quote(quote) == (False, reason)


def test_state_screens_are_keyed_by_state_name():
    assert STATE_SCREENS == {
        METASTATIC: screen_metastatic_quote,
        CASTRATION_RESISTANT: screen_castration_resistant_quote,
    }
