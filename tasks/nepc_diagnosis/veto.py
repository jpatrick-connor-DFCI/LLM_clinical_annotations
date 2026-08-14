"""Deterministic quote-level gate for the strict NEPC diagnosis task.

Applied AFTER the quote is grounded in evidence, and the model cannot override
it. Precision is the objective here: an ambiguous quote is rejected, and a
rejection costs one finding rather than the patient's whole result.

This exists because the recall-biased longitudinal pipeline reported three
classes of false positive that prompt wording alone did not prevent:
negated statements read as positive, IHC results reported as a diagnosis, and a
single isolated term mention treated as a diagnosis. Each maps to one check
below, as does a fourth class found later: clinical-trial eligibility
boilerplate, which describes an NEPC population the patient is being screened
against rather than a diagnosis the patient has.

The gate accepts pathology and clinician wording equally -- for many patients
the diagnosis is recorded only in an oncology progress note. Clinical assertion
contexts ("Assessment:", "Problem list:", "s/p") anchor a diagnosis just as a
pathology diagnosis line does, and the disease acronyms are self-anchoring. The
negation, hedge, and surveillance cues apply identically to both sources.

Bump VETO_VERSION on any change to the patterns or windows. The stage-2 run
fingerprint hashes it, so a changed gate forces --overwrite instead of silently
mixing labels adjudicated under two different gates.
"""

import re

VETO_VERSION = "nepc-dx-veto-v3"

# The disease term whose assertion status is being adjudicated.
NEPC_TERM = re.compile(
    r"\b(?:small[\s-]?cell|oat[\s-]?cell|nepc|scpc|scnc|neuro[\s-]?endocrine)\b",
    flags=re.IGNORECASE,
)

# Disease acronyms that are self-anchoring: each already denotes a specific
# carcinoma ("NEPC" = neuroendocrine prostate cancer, "SCPC" = small cell
# prostate carcinoma), so requiring a separate "carcinoma"/"cancer" word next to
# them is redundant. Clinicians writing progress notes use the bare acronym
# constantly ("Assessment: NEPC, on treatment"), and demanding a spelled-out
# noun would drop those genuine diagnoses. The spelled-out terms ("small cell",
# "neuroendocrine") are NOT self-anchoring -- they are adjectives that need a
# noun, which is exactly what separates a diagnosis from "neuroendocrine
# features".
SELF_ANCHORING_TERM = re.compile(
    r"\b(?:t[\s-]?nepc|nepc|scpc|scnc)\b",
    flags=re.IGNORECASE,
)

# A diagnostic assertion must appear in the SAME quote. "Positive for
# synaptophysin", "shows neuroendocrine features", and a bare term mention all
# fail this check -- exactly the observed false-positive classes.
#
# Includes clinical assertion contexts alongside pathology ones: an oncologist's
# "Assessment:"/"Impression:"/"Problem list:" line states the patient's
# diagnosis just as authoritatively as a pathology diagnosis line, and for many
# patients the progress note is where the diagnosis is recorded.
ASSERTION_ANCHOR = re.compile(
    r"\b(?:diagnos(?:is|es|ed|tic)|dx|final\s+diagnosis|"
    r"pathologic(?:al)?\s+diagnosis|consistent\s+with|compatible\s+with|c/w|"
    r"biops(?:y|ies)\s+(?:showed|showing|revealed|demonstrated|confirmed)|"
    r"transform(?:ation|ed)|transdifferentiat\w*|"
    r"proven|confirmed|established|known|"
    r"carcinoma|cancer|malignancy|tumou?r|"
    r"assessment|impression|problem\s+list|active\s+problem|"
    r"status\s+post|s/p|on\s+treatment\s+for|being\s+treated\s+for|"
    r"history\s+of)\b",
    flags=re.IGNORECASE,
)

# Negation, hedging, and non-assertion cues. Scanned in a bounded window BEFORE
# each disease term, never across the whole quote -- a whole-quote scan would
# veto "Final diagnosis: small cell carcinoma; no evidence of nodal disease".
#
# Deliberately NOT cues: "features" and "differentiation". Requiring an
# assertion anchor already excludes "neuroendocrine features" standing alone,
# and vetoing on them would kill valid quotes such as "small cell carcinoma of
# the prostate with neuroendocrine features".
#
# Non-prostate primary sites are handled separately by NON_PROSTATE_SITE below,
# which scans both directions -- the site often follows the term ("small cell
# carcinoma of the lung") rather than preceding it.
NEGATION_HEDGE = re.compile(
    r"\b(?:no|not|non|negative|neg\.?|without|absent|absence|denies|denied|"
    r"rule[\s-]?out|r/o|ruled\s+out|exclude[ds]?|excluding|"
    r"versus|vs\.?|differential|ddx|consider(?:ed|ation)?|"
    r"suspicious|suspect(?:ed)?|suspicion|concern(?:ing|ed)?|"
    r"possible|possibly|probable|probably|likely|unlikely|"
    r"cannot\s+(?:be\s+)?exclude[d]?|equivocal|indeterminate|questionable|"
    r"favor(?:s|ed|ing)?|worrisome|atypical|"
    r"if|should|would|may|might|could|"
    r"screen(?:ing)?|evaluate|assess|work[\s-]?up|"
    r"monitor(?:ing|ed)?|surveillance|watch(?:ing|ed)?\s+for|"
    r"develops?|develop(?:ing|ed)|progress(?:es|ion)\s+to|"
    r"repeat|pending|await(?:ing|ed)?|risk\s+of|potential|"
    r"family\s+history|mother|father|brother|sister)\b",
    flags=re.IGNORECASE,
)

# Postposed negation: "small cell carcinoma is not present". A few noun words
# ("carcinoma", "of the prostate") may intervene between the term and the
# copula, so allow a short run of them rather than anchoring immediately.
POST_NEGATION = re.compile(
    r"^(?:\W+|\w+\s+){0,4}?(?:is|was|are|were)\s+(?:not|negative|absent|"
    r"ruled\s+out|excluded|unlikely)\b",
    flags=re.IGNORECASE,
)

# Non-prostate primary sites. Scanned in a window on BOTH sides of the term:
# "small cell carcinoma of the lung" puts the site after the term, while
# "lung primary with small cell histology" puts it before. Small-cell lung
# carcinoma is the single largest false-positive source for this trigger set.
NON_PROSTATE_SITE = re.compile(
    r"\b(?:lung|pulmonary|bronch\w*|thoracic|bladder|urothelial|"
    r"gastrointestinal|pancrea\w*|esophag\w*|colon|rectal|cervi\w*|"
    r"ovarian|breast|merkel|skin|cutaneous)\b",
    flags=re.IGNORECASE,
)
SITE_WINDOW_CHARS = 60

# Stock text: clinical-trial eligibility criteria, protocol titles, consent
# language, and registry/education boilerplate. These describe a POPULATION the
# patient is being screened against, not a diagnosis the patient has -- and they
# select for NEPC wording precisely because the trial targets NEPC, so they are a
# systematic false-positive source rather than a random one. They are also
# copy-forwarded across many notes, so one boilerplate block inflates a
# patient's candidate count out of proportion to the real evidence.
#
# Scanned on BOTH sides of the term, like NON_PROSTATE_SITE: "inclusion
# criteria: patients with small cell carcinoma" puts the cue before,
# "...small cell carcinoma are eligible for enrollment" puts it after.
#
# Scoped to a window around the term, NOT the whole quote or the whole note: a
# pathology diagnosis and a trial discussion routinely appear in the same note,
# and the diagnosis is often exactly WHY the trial is being discussed. Vetoing
# on any protocol mention anywhere would discard genuine diagnoses.
BOILERPLATE = re.compile(
    r"\b(?:inclusion|exclusion|eligib\w*|ineligible|enroll\w*|"
    r"protocol|trial|study\s+(?:of|in|population|arm|drug)|"
    r"phase\s+(?:i{1,3}|1|2|3|iv|4)\b|nct\d*|cohort\s+[a-z0-9]\b|"
    r"consent(?:ed|ing)?|screening\s+(?:log|criteria)|randomiz\w*|"
    r"registry|questionnaire|"
    r"subjects?\s+must|patients?\s+must|candidates?\s+for)\b",
    flags=re.IGNORECASE,
)
BOILERPLATE_WINDOW_CHARS = 100

# Characters of the flattened quote. ~120 chars ~= 20 tokens: long enough to
# catch "there is no evidence of an underlying ...", short enough that an
# unrelated earlier clause does not veto a clean assertion later in the quote.
NEGATION_WINDOW_CHARS = 120
POST_WINDOW_CHARS = 40


def screen_quote(quote):
    """Return ``(ok, reason)`` for one grounded quote.

    Reasons are stable strings and land in the rejected-findings audit, so the
    reason histogram is the tuning signal for these patterns:
      ``no_nepc_term`` / ``no_diagnostic_assertion`` /
      ``negated_or_hedged:<cue>`` / ``boilerplate:<cue>``
    """
    text = re.sub(r"\s+", " ", str(quote or "")).strip()
    if not text:
        return False, "no_nepc_term"
    hits = list(NEPC_TERM.finditer(text))
    if not hits:
        return False, "no_nepc_term"
    # A self-anchoring acronym IS the diagnosis; anything else needs a separate
    # assertion in the same quote.
    if (
        SELF_ANCHORING_TERM.search(text) is None
        and ASSERTION_ANCHOR.search(text) is None
    ):
        return False, "no_diagnostic_assertion"
    # Every occurrence must be clean: one negated mention anywhere in the quote
    # makes it an unsafe basis for a strict-precision positive.
    for hit in hits:
        before = text[max(0, hit.start() - NEGATION_WINDOW_CHARS):hit.start()]
        cue = NEGATION_HEDGE.search(before)
        if cue is not None:
            return False, f"negated_or_hedged:{cue.group(0).lower()}"
        after = text[hit.end():hit.end() + POST_WINDOW_CHARS]
        if POST_NEGATION.match(after):
            return False, "negated_or_hedged:post_negation"
        # A non-prostate primary named on either side of the term attributes the
        # neuroendocrine histology to a different cancer.
        site_window = text[
            max(0, hit.start() - SITE_WINDOW_CHARS):hit.end() + SITE_WINDOW_CHARS
        ]
        site = NON_PROSTATE_SITE.search(site_window)
        if site is not None:
            return False, f"negated_or_hedged:{site.group(0).lower()}"
        # Stock trial/protocol text describes an eligible population, not this
        # patient's diagnosis. Its own reason prefix keeps it separately
        # tunable in the rejection histogram.
        boiler_window = text[
            max(0, hit.start() - BOILERPLATE_WINDOW_CHARS):
            hit.end() + BOILERPLATE_WINDOW_CHARS
        ]
        boiler = BOILERPLATE.search(boiler_window)
        if boiler is not None:
            return False, f"boilerplate:{boiler.group(0).lower()}"
    return True, None
