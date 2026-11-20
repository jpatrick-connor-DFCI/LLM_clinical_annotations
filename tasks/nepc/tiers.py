"""Four pathology-based groups derived from gated findings.

Group 1 includes definitive NE carcinoma and mixed tumors with a confirmed
NE carcinoma component. Group 2 requires explicit poor/dedifferentiation and
NE features. Group 3 requires an explicit adenocarcinoma diagnosis and NE
features or positive tissue markers; Gleason grade alone never sets group 2.
Group 4 requires adenocarcinoma, no gated NE finding, and completed review.
Cohort membership supplies clinical eligibility. Missing evidence is not a
reference negative. Pathology reports and explicit pathology summaries count.

The t1/t2/t3 flags can record different events in a patient's history;
`nepc_tier` selects t1 > t2 > t3 > t4. Group 4 is assigned at patient rollup,
not to an isolated negative report in a patient with NE evidence elsewhere.
"""

import re

from tasks.nepc.ne_gate import (
    ADENO_TERM, MIXED_CUE, NE_TERM, PD_CARCINOMA, screen_asserted, screen_positive_ihc,
)
from tasks.ihc_polarity import flatten
from tasks.onset import both_hold, date_interval

TIER_VERSION = "nepc-tier-v3"
TIERS = ("t1", "t2", "t3", "t4")

PD_IHC_PAIR = "poorly_differentiated_carcinoma"
ADENO = "prostate_adenocarcinoma"
IHC = "positive_ne_ihc"
NON_NE_TYPES = {PD_IHC_PAIR, ADENO}

DEFINITIVE_NE = re.compile(
    r"\b(?:small[\s-]?cell|oat[\s-]?cell)(?:\s+neuro[\s-]?endocrine)?\s+carcinoma\b|"
    r"\b(?:large[\s-]?cell|high[\s-]+grade)\s+neuro[\s-]?endocrine\s+carcinoma\b",
    re.IGNORECASE,
)
NE_CARCINOMA = re.compile(r"\bneuro[\s-]?endocrine\s+carcinoma\b", re.IGNORECASE)
# The quote itself must link a pathology source to an established result;
# an assessment/problem-list diagnosis or treatment choice alone is insufficient.
PATHOLOGY_RESULT = re.compile(
    r"\b(?:biopsy|biopsies|pathology|histology|histopathology|resection|specimen|turp)\b"
    r"[^.;!?]{0,100}?\b(?:show(?:s|ed|ing)|reveal(?:s|ed)|confirm(?:s|ed)|"
    r"demonstrat(?:es|ed)|diagnos(?:ed|tic)|report(?:s|ed)|consistent\s+with)\b|"
    r"\b(?:biopsy|histologically|pathologically)[\s-]+(?:proven|confirmed)\b|"
    r"\b(?:biopsy|pathology|histology)\s*:",
    re.IGNORECASE,
)


def pathology_supported(row):
    return row.get("note_type") == "Pathology" or (
        row.get("note_type") == "Clinician"
        and screen_asserted(flatten(row["quote"]), PATHOLOGY_RESULT)[0]
    )


def asserted(text, term):
    return screen_asserted(text, term)[0]


def has_ne_evidence(row):
    """Histology support can also contain an explicitly asserted NE finding."""
    text = flatten(row["quote"])
    return (row["evidence_type"] not in NON_NE_TYPES
            or asserted(text, NE_TERM) or screen_positive_ihc(text)[0])


def finding_tiers(row):
    """Groups supported by one gated, pathology-supported finding."""
    if not pathology_supported(row):
        return set()
    kind, text = row["evidence_type"], flatten(row["quote"])
    if not has_ne_evidence(row):
        return set()
    if asserted(text, DEFINITIVE_NE):
        return {"t1"}
    if (asserted(text, NE_CARCINOMA) and asserted(text, ADENO_TERM)
            and MIXED_CUE.search(text)):
        return {"t1"}
    # A transformation/NEPC assertion without a definitive carcinoma target
    # cannot establish group 1. It may still describe a group 2 or 3 phenotype.
    if kind in {"pd_carcinoma_ne_features", "ne_differentiation", IHC,
                "histologic_transformation", ADENO, PD_IHC_PAIR}:
        if asserted(text, PD_CARCINOMA):
            return {"t2"}
        if asserted(text, ADENO_TERM):
            return {"t3"}
    return set()


def _compatible(left, right):
    """Do not pair explicitly different specimens in one report."""
    a, b = left.get("specimen_site"), right.get("specimen_site")
    if a and b and flatten(a).lower() != flatten(b).lower():
        return False
    # A summary can contain several biopsies of the same organ. Disjoint
    # event intervals cannot establish a histology/IHC pair for one specimen.
    dates = [date_interval(r.get("event_date"), r.get("date_precision"))
             for r in (left, right)]
    return (None in dates or
            max(d[0] for d in dates) <= min(d[1] for d in dates))


def _pair(diagnosis, ne):
    """A combination starts when both components hold, never at the older one.

    Keep the original quotes separately so the derived evidence is not presented
    as one contiguous source quotation. Undated components leave onset unknown.
    """
    onset = both_hold([diagnosis, ne])
    return {
        **ne,
        **(onset or {"event_date": None, "date_precision": "unknown"}),
        "evidence_type": diagnosis["evidence_type"] + "+" + ne["evidence_type"],
        "supporting_quotes": [diagnosis["quote"], ne["quote"]],
    }


def patient_tiers(rows, *, complete=False):
    """Return supporting rows per group; only complete review can set group 4."""
    by_note = {}
    for row in rows:
        if pathology_supported(row):
            by_note.setdefault(row["note_uid"], []).append(row)
    out = {tier: [] for tier in TIERS}
    adeno_rows = []
    for note_rows in by_note.values():
        own = [(row, finding_tiers(row)) for row in note_rows]
        for row, tiers in own:
            for tier in tiers:
                out[tier].append(row)
        adeno = [r for r in note_rows if r["evidence_type"] == ADENO]
        adeno_rows.extend(adeno)
        pd = [r for r in note_rows if r["evidence_type"] == PD_IHC_PAIR]
        ne = [r for r in note_rows if r["evidence_type"] in {IHC, "ne_differentiation"}]
        for feature in ne:
            # A separately extracted IHC panel is explained by the diagnosis
            # only for the same dated event and a compatible specimen.
            explained = any(
                tiers & {"t1", "t2"} and _compatible(row, feature)
                and row["event_date"] == feature["event_date"]
                for row, tiers in own
            )
            if explained or finding_tiers(feature):
                continue
            partners = [r for r in pd if _compatible(r, feature)]
            tier = "t2"
            if not partners:
                partners = [r for r in adeno if _compatible(r, feature)]
                tier = "t3"
            out[tier].extend(_pair(r, feature) for r in partners)
    # A lone positive marker or unsupported clinician NE diagnosis blocks the
    # reference group even when it cannot establish any of groups 1-3.
    if (complete and not any(out[t] for t in ("t1", "t2", "t3"))
            and not any(has_ne_evidence(r) for r in rows)):
        out["t4"].extend(adeno_rows)
    return out


def highest_tier(tier_rows):
    return next((tier for tier in TIERS if tier_rows.get(tier)), None)
