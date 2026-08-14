"""Deterministic quote-level gate for the strict NEPC diagnosis task.

Applied AFTER the quote is grounded in evidence, and the model cannot override
it. Precision is the objective here: an ambiguous quote is rejected, and a
rejection costs one finding rather than the patient's whole result.

This exists because the recall-biased longitudinal pipeline reported three
classes of false positive that prompt wording alone did not prevent:
negated statements read as positive, IHC results reported as a diagnosis, and a
single isolated term mention treated as a diagnosis. Each maps to one check
below.

Bump VETO_VERSION on any change to the patterns or windows. The stage-2 run
fingerprint hashes it, so a changed gate forces --overwrite instead of silently
mixing labels adjudicated under two different gates.
"""

import re

VETO_VERSION = "nepc-dx-veto-v1"

# The disease term whose assertion status is being adjudicated.
NEPC_TERM = re.compile(
    r"\b(?:small[\s-]?cell|oat[\s-]?cell|nepc|scpc|scnc|neuro[\s-]?endocrine)\b",
    flags=re.IGNORECASE,
)

# A diagnostic assertion must appear in the SAME quote. "Positive for
# synaptophysin", "shows neuroendocrine features", and a bare term mention all
# fail this check -- exactly the observed false-positive classes.
ASSERTION_ANCHOR = re.compile(
    r"\b(?:diagnos(?:is|es|ed|tic)|dx|final\s+diagnosis|"
    r"pathologic(?:al)?\s+diagnosis|consistent\s+with|compatible\s+with|c/w|"
    r"biops(?:y|ies)\s+(?:showed|showing|revealed|demonstrated|confirmed)|"
    r"transform(?:ation|ed)|transdifferentiat\w*|"
    r"proven|confirmed|established|known|"
    r"carcinoma|cancer|malignancy|tumou?r|"
    r"history\s+of|status\s+post)\b",
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

# Characters of the flattened quote. ~120 chars ~= 20 tokens: long enough to
# catch "there is no evidence of an underlying ...", short enough that an
# unrelated earlier clause does not veto a clean assertion later in the quote.
NEGATION_WINDOW_CHARS = 120
POST_WINDOW_CHARS = 40


def screen_quote(quote):
    """Return ``(ok, reason)`` for one grounded quote.

    Reasons are stable strings and land in the rejected-findings audit, so the
    reason histogram is the tuning signal for these patterns:
      ``no_nepc_term`` / ``no_diagnostic_assertion`` / ``negated_or_hedged:<cue>``
    """
    text = re.sub(r"\s+", " ", str(quote or "")).strip()
    if not text:
        return False, "no_nepc_term"
    hits = list(NEPC_TERM.finditer(text))
    if not hits:
        return False, "no_nepc_term"
    if ASSERTION_ANCHOR.search(text) is None:
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
    return True, None
