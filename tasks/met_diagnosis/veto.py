"""Deterministic quote-level gate for the metastatic prostate cancer task.

Applied AFTER the quote is grounded in evidence, and the model cannot override
it. Precision is the objective here: an ambiguous quote is rejected, and a
rejection costs one finding rather than the patient's whole result.

This is the sibling of tasks/nepc_diagnosis/veto.py and keeps its shape --
flatten, require a term, require an assertion anchor, then require EVERY term
occurrence to survive a set of windowed checks. Three things differ, and each
is a place where copying the NEPC gate verbatim would be wrong:

  1. Negation is the dominant case, not an edge case. The single most common
     metastasis sentence in this corpus is "no evidence of metastatic disease",
     and a surveillance imaging report repeats some variant of it every few
     months for years. The NEGATION_HEDGE set is therefore extended with the
     radiology vocabulary that carries the same meaning without the word "no"
     (degenerative, benign, bone island, age-related, postsurgical).

  2. A non-prostate site near the term is usually a DESTINATION, not a primary.
     In the NEPC task "small cell carcinoma of the lung" always meant a lung
     primary, so any nearby site was disqualifying. Here "metastatic prostate
     cancer to the lung" is precisely the finding we want. The site check
     therefore fires only when the site reads as the primary
     (NON_PROSTATE_PRIMARY_CUE) rather than merely appearing nearby.

  3. Regional pelvic nodal disease is N1, not M1, and is excluded by the task
     definition -- but it is written with the word "metastatic" ("metastatic
     adenocarcinoma in 2 of 14 pelvic lymph nodes"). This needs a check the
     NEPC gate has no analogue for, and it is a property of the quote's whole
     site set rather than of one term occurrence, so it runs after the loop.

Bump VETO_VERSION on any change to the patterns or windows. The stage-2 run
fingerprint hashes it, so a changed gate forces --overwrite instead of silently
mixing labels adjudicated under two different gates.
"""

import re

VETO_VERSION = "met-dx-veto-v2"

# The term whose assertion status is being adjudicated. Includes bare lesion
# descriptors, which the collector retrieves and which can be genuine first
# documentation of bone metastasis ("innumerable sclerotic osseous lesions"),
# but which are NOT self-anchoring -- they need a separate assertion.
MET_TERM = re.compile(
    r"(?:\b|(?<=\d))(?:metasta\w*|mets|m1[abc]?|stage\s+(?:iv|4)|carcinomatosis|"
    r"disseminated\s+disease|"
    r"(?:osseous|skeletal|bone)\s+(?:lesions?|involvement|disease|uptake)|"
    r"(?:sclerotic|lytic)\s+(?:bone\s+)?lesions?)\b",
    flags=re.IGNORECASE,
)

# Terms that already assert distant spread on their own. "Metastatic" is an
# assertion of spread in a way that "osseous lesions" is not, and the disease
# acronyms (mHSPC/mCRPC) and M/stage categories encode it by definition.
# Clinicians write these bare constantly ("Assessment: mCRPC, on lutetium"), and
# demanding a separate anchor would drop those genuine assertions.
SELF_ANCHORING_TERM = re.compile(
    r"(?:\b|(?<=\d))(?:metastati\w*|metastas[ei]s|metastasis|mets|m1[abc]?|"
    r"stage\s+(?:iv|4)|mhspc|mcrpc|carcinomatosis|disseminated\s+disease)\b",
    flags=re.IGNORECASE,
)

# A malignant assertion must appear in the SAME quote for the non-self-anchoring
# lesion descriptors. "Sclerotic lesions in the pelvis" alone fails this; "...
# consistent with osseous metastases" passes on the term itself.
ASSERTION_ANCHOR = re.compile(
    r"\b(?:metasta\w*|malignan\w*|carcinoma|cancer|tumou?r|neoplas\w*|"
    r"diagnos(?:is|es|ed|tic)|dx|consistent\s+with|compatible\s+with|c/w|"
    r"biops(?:y|ies)\s+(?:showed|showing|revealed|demonstrated|confirmed)|"
    r"proven|confirmed|established|known|"
    r"assessment|impression|problem\s+list|active\s+problem|"
    r"status\s+post|s/p|on\s+treatment\s+for|being\s+treated\s+for|"
    r"history\s+of)\b",
    flags=re.IGNORECASE,
)

# Negation, hedging, and non-assertion cues. Scanned in a bounded window BEFORE
# each term, never across the whole quote -- a whole-quote scan would veto
# "IMPRESSION: osseous metastases; no evidence of visceral disease".
#
# Extended beyond the NEPC gate with radiology-specific alternatives that negate
# malignancy without negating the noun: a sclerotic focus called "degenerative"
# or "a bone island" is an explicit statement that the lesion is NOT a
# metastasis. "Resolved"/"no new" likewise describe absence at this timepoint.
NEGATION_HEDGE = re.compile(
    r"\b(?:no|not|non|negative|neg\.?|without|absent|absence|denies|denied|"
    r"free\s+of|clear\s+of|"
    r"rule[\s-]?out|r/o|ruled\s+out|exclude[ds]?|excluding|"
    r"versus|vs\.?|differential|ddx|consider(?:ed|ation)?|"
    r"suspicious|suspect(?:ed)?|suspicion|concern(?:ing|ed)?|"
    r"possible|possibly|probable|probably|likely|unlikely|"
    r"cannot\s+(?:be\s+)?exclude[d]?|equivocal|indeterminate|questionable|"
    r"favor(?:s|ed|ing)?|worrisome|atypical|"
    r"degenerative|arthriti\w*|osteoarthriti\w*|bone\s+island|"
    r"benign|age[\s-]related|postsurgical|post[\s-]operative|"
    r"healing|fracture|traumatic|infectious|inflammatory|"
    r"attributable\s+to|felt\s+to\s+be|thought\s+to\s+(?:be|represent)|"
    r"no\s+new|resolved|"
    r"if|should|would|may|might|could|"
    r"screen(?:ing)?|evaluate|assess|work[\s-]?up|restag\w*|"
    r"monitor(?:ing|ed)?|surveillance|watch(?:ing|ed)?\s+for|"
    r"develops?|develop(?:ing|ed)|progress(?:es|ion)\s+to|"
    r"repeat|pending|await(?:ing|ed)?|risk\s+(?:of|for)|at\s+risk|potential|"
    r"family\s+history|mother|father|brother|sister)\b",
    flags=re.IGNORECASE,
)

# Copy-forward status words. These are NOT unconditional negation cues:
# "stable metastatic disease on abiraterone" is a genuine positive and one of
# the most common ways established metastatic disease is written in an oncology
# note. They negate only in the "stable, no metastatic disease" shape, which the
# NEGATION_HEDGE "no" already catches -- so they are listed here purely so that
# STATUS_WORD_ONLY can be checked separately and not fold into the hedge set.
# Kept as an explicit named pattern so a future tuning pass has somewhere
# obvious to put a real rule if the histogram shows one is needed.
STATUS_WORD = re.compile(r"\b(?:stable|unchanged|similar|comparable)\b", flags=re.IGNORECASE)

# Postposed negation: "osseous metastases are not present", "metastatic disease
# was excluded". A few noun words may intervene between the term and the copula,
# so allow a short run of them rather than anchoring immediately.
POST_NEGATION = re.compile(
    r"^(?:\W+|\w+\s+){0,4}?(?:is|was|are|were)\s+(?:not|negative|absent|"
    r"ruled\s+out|excluded|unlikely|resolved)\b",
    flags=re.IGNORECASE,
)

# "negative for metastatic disease" / "negative for osseous metastases" -- the
# cue FOLLOWS nothing and PRECEDES the term at a distance the 120-char window
# would catch, but the idiom is common enough in bone-scan impressions to be
# worth its own pattern so its rejections are legible in the histogram.
NEGATIVE_FOR = re.compile(r"\bnegative\s+for\b", flags=re.IGNORECASE)

# Non-prostate primary sites. Unlike the NEPC gate, a bare site near the term is
# NOT disqualifying here -- it is usually the metastatic destination. The site
# vetoes only when a primary-attribution cue sits next to it.
NON_PROSTATE_SITE = re.compile(
    r"\b(?:lung|pulmonary|bronch\w*|bladder|urothelial|"
    r"gastrointestinal|pancrea\w*|esophag\w*|colon|colorectal|rectal|gastric|"
    r"renal|kidney|melanoma|lymphoma|leukemi\w*|"
    r"cervi\w*|ovarian|breast|merkel|thyroid|hepatocellular)\b",
    flags=re.IGNORECASE,
)
NON_PROSTATE_PRIMARY_CUE = re.compile(
    r"\b(?:primary|primaries|origin|originating|arising|"
    r"known|history\s+of|h/o|s/p|status\s+post|"
    r"derived|consistent\s+with|compatible\s+with|"
    r"carcinoma|cancer|adenocarcinoma|malignancy)\b",
    flags=re.IGNORECASE,
)
SITE_WINDOW_CHARS = 60
PRIMARY_CUE_WINDOW_CHARS = 30

# Prostate-attribution cues. When the quote independently names prostate as the
# disease, a nearby non-prostate site is a destination, not a competing primary,
# so the primary check is skipped. "Metastatic prostate cancer to the liver".
PROSTATE_CUE = re.compile(
    r"\b(?:prostate|prostatic|psa|mhspc|mcrpc|crpc|"
    r"castration[\s-]resistant|hormone[\s-]sensitive)\b",
    flags=re.IGNORECASE,
)

# Regional (N1) nodal stations. Prostate cancer confined to these is N1, not M1.
REGIONAL_NODE = re.compile(
    r"\b(?:pelvic|obturator|hypogastric|perirectal|periprostatic|"
    r"(?:internal|external)\s+iliac|iliac|regional)\b",
    flags=re.IGNORECASE,
)

# Distant (M1a) nodal stations, plus every non-nodal distant site. Presence of
# any of these anywhere in the quote means the quote is not regional-nodal-only.
DISTANT_SITE = re.compile(
    r"\b(?:retroperitoneal|para[\s-]?aortic|paraaortic|mediastinal|"
    r"supraclavicular|cervical\s+node|inguinal|axillary|"
    r"bone|osseous|skeletal|spine|spinal|vertebr\w*|rib|femur|femoral|"
    r"pelvis|sacrum|ilium|humerus|sternum|scapula|skull|calvari\w*|"
    r"liver|hepatic|lung|pulmonary|adrenal|brain|cerebral|"
    r"pleural|peritoneal|soft\s+tissue|"
    r"m1[bc]|visceral|widespread|diffuse|innumerable|multiple)\b",
    flags=re.IGNORECASE,
)

# Nodal-disease vocabulary. Used only to decide whether a quote is ABOUT nodes.
NODAL_TERM = re.compile(
    r"\b(?:lymph\s*node[s]?|nodal|node[s]?|lymphadenopathy|adenopathy|"
    r"n1|lymph)\b",
    flags=re.IGNORECASE,
)

# Stock text: clinical-trial eligibility criteria, protocol titles, consent
# language, and registry/education boilerplate. These describe a POPULATION the
# patient is being screened against, not the patient's disease -- and they
# select for metastatic wording precisely because such trials enroll metastatic
# patients, so they are a systematic false-positive source. Scoped to a window
# around the term, not the whole quote: a real assessment and a trial discussion
# routinely appear in the same note, and the disease is often exactly WHY the
# trial is being discussed.
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


def _is_non_prostate_primary(window):
    """True when a non-prostate site in `window` reads as the PRIMARY cancer.

    A site alone is not enough -- in this task the site is usually the
    metastatic destination. It disqualifies only when an attribution cue
    ("known", "primary", "history of", "carcinoma") sits within a short window
    of the site itself.
    """
    for site in NON_PROSTATE_SITE.finditer(window):
        cue_window = window[
            max(0, site.start() - PRIMARY_CUE_WINDOW_CHARS):
            site.end() + PRIMARY_CUE_WINDOW_CHARS
        ]
        if NON_PROSTATE_PRIMARY_CUE.search(cue_window):
            return site.group(0).lower()
    return None


def _is_regional_nodal_only(text):
    """True when the quote's only asserted site is a regional (N1) node station.

    Runs on the whole quote rather than per-occurrence: "metastatic
    adenocarcinoma in 2 of 14 pelvic lymph nodes" is disqualified by what the
    quote as a whole does and does not name, not by any single term hit.
    """
    if not NODAL_TERM.search(text):
        return False
    if not REGIONAL_NODE.search(text):
        return False
    # Any distant site named anywhere in the quote means this is not
    # regional-only -- "pelvic and retroperitoneal adenopathy" qualifies.
    if DISTANT_SITE.search(text):
        return False
    return True


def screen_quote(quote):
    """Return ``(ok, reason)`` for one grounded quote.

    Reasons are stable strings and land in the rejected-findings audit, so the
    reason histogram is the tuning signal for these patterns:
      ``no_met_term`` / ``no_metastatic_assertion`` /
      ``negated_or_hedged:<cue>`` / ``regional_nodal_only`` /
      ``non_prostate_primary:<cue>`` / ``boilerplate:<cue>``
    """
    text = re.sub(r"\s+", " ", str(quote or "")).strip()
    if not text:
        return False, "no_met_term"
    hits = list(MET_TERM.finditer(text))
    if not hits:
        return False, "no_met_term"
    # A self-anchoring term IS the assertion of spread; a bare lesion descriptor
    # needs a separate malignant assertion in the same quote.
    if (
        SELF_ANCHORING_TERM.search(text) is None
        and ASSERTION_ANCHOR.search(text) is None
    ):
        return False, "no_metastatic_assertion"
    # "negative for metastatic disease" is idiomatic enough to check up front,
    # so its rejections are legible rather than folded into a generic cue.
    if NEGATIVE_FOR.search(text):
        return False, "negated_or_hedged:negative for"
    # Every occurrence must be clean: one negated mention anywhere in the quote
    # makes it an unsafe basis for a strict-precision positive.
    prostate_named = PROSTATE_CUE.search(text) is not None
    for hit in hits:
        before = text[max(0, hit.start() - NEGATION_WINDOW_CHARS):hit.start()]
        cue = NEGATION_HEDGE.search(before)
        if cue is not None:
            return False, f"negated_or_hedged:{cue.group(0).lower()}"
        after = text[hit.end():hit.end() + POST_WINDOW_CHARS]
        if POST_NEGATION.match(after):
            return False, "negated_or_hedged:post_negation"
        # A non-prostate site is only disqualifying when it reads as the
        # primary. Skipped entirely when the quote independently names prostate
        # disease, since then the site is a destination.
        if not prostate_named:
            site_window = text[
                max(0, hit.start() - SITE_WINDOW_CHARS):hit.end() + SITE_WINDOW_CHARS
            ]
            site = _is_non_prostate_primary(site_window)
            if site is not None:
                return False, f"non_prostate_primary:{site}"
        # Stock trial/protocol text describes an eligible population, not this
        # patient's disease. Its own reason prefix keeps it separately tunable.
        boiler_window = text[
            max(0, hit.start() - BOILERPLATE_WINDOW_CHARS):
            hit.end() + BOILERPLATE_WINDOW_CHARS
        ]
        boiler = BOILERPLATE.search(boiler_window)
        if boiler is not None:
            return False, f"boilerplate:{boiler.group(0).lower()}"
    # Whole-quote check: N1 disease is written with the word "metastatic" but is
    # not M1, and the task definition excludes it.
    if _is_regional_nodal_only(text):
        return False, "regional_nodal_only"
    return True, None
