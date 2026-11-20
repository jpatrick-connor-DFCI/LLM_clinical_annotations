"""Deterministic quote gate for neuroendocrine findings.

Applied at label-build time to every grounded NEPC finding, and reused by the
AVPC rollup for C1 (small-cell histology) and C6's tissue marker. The model
cannot override it. Precision is the objective: an ambiguous quote is
rejected, which costs one finding rather than the patient's label.

It follows `tasks/met_veto.py`'s conventions: flatten,
require the term the evidence type claims, require EVERY term occurrence to
survive bounded negation/hedge, post-negation, boilerplate and
non-prostate-primary checks, and return stable reason strings for the audit.

Two departures from the met gate:

  - The negation window stops at the previous sentence boundary. Pathology
    quotes pack many short negative sentences together ("No perineural
    invasion. Small cell carcinoma, extensive."), and a cue from an earlier
    sentence is not about this term.
  - IHC is judged per marker, by the polarity cue nearest to it. Panels
    routinely mix results ("positive for synaptophysin and negative for
    chromogranin"), so one negative marker must not veto a positive one, and
    "negative for synaptophysin, chromogranin and CD56" must reject all three.

Bump NE_GATE_VERSION on any change to these patterns or windows. The label
metadata records it, and --rebuild-labels-only re-applies a changed gate to
stored findings without new LLM calls.
"""

import re

from tasks.met_veto import (
    NEGATION_WINDOW_CHARS,
    POST_WINDOW_CHARS,
    PROSTATE_CUE,
    SITE_WINDOW_CHARS,
    _is_non_prostate_primary,
)
from tasks.ihc_polarity import (  # noqa: F401  (re-exported for callers)
    _BINDS_FORWARD,
    IHC_ASSERTION_NEGATION,
    IHC_CLAUSE_BREAK,
    IHC_NEGATIVE,
    IHC_POLARITY_POST_CHARS,
    IHC_POLARITY_PRE_CHARS,
    IHC_POSITIVE,
    IHC_POST_NEGATION,
    IHC_RESULT_NOUN,
)
from tasks.ihc_polarity import SENTENCE_END as _SENTENCE_END  # noqa: F401
from tasks.ihc_polarity import flatten as _flatten
from tasks.ihc_polarity import nearest_polarity as _nearest_polarity
from tasks.ihc_polarity import positive_cue_is_negated as _positive_cue_is_negated  # noqa: F401
from tasks.ihc_polarity import same_sentence_after as _same_sentence_after
from tasks.ihc_polarity import same_sentence_before as _same_sentence_before
from tasks.trial_guard import trial_boilerplate_reason

NE_GATE_VERSION = "ne-gate-v7"

# Any neuroendocrine/small-cell term. "non-small cell" is caught separately.
NE_TERM = re.compile(
    r"\b(?:neuro[\s-]?endocrine|t-?nepc|nepc|small[\s-]?cell|scpc|scnc|"
    r"oat[\s-]?cell)\b",
    flags=re.IGNORECASE,
)
SMALL_CELL_TERM = re.compile(
    r"\b(?:small[\s-]?cell|scpc|scnc|oat[\s-]?cell)\b", flags=re.IGNORECASE
)
# A diagnosis: an NE term naming a carcinoma (or its component), or an acronym
# that is itself a diagnosis.
NE_DIAGNOSIS = re.compile(
    r"\b(?:t-?nepc|nepc|scpc|scnc)\b|"
    r"\b(?:neuro[\s-]?endocrine|small[\s-]?cell|oat[\s-]?cell)"
    r"(?:\s+[\w/()-]+){0,4}?\s+"
    r"(?:carcinoma|cancer|tumou?r|neoplasm|malignancy|component|histology|"
    r"variant|subtype|type|disease|prostate)\b|"
    r"\b(?:carcinoma|cancer)\s+(?:with|of)\s+(?:\w+\s+){0,2}?"
    r"(?:neuro[\s-]?endocrine|small[\s-]?cell)\b",
    flags=re.IGNORECASE,
)
TRANSFORMATION = re.compile(
    r"\b(?:transform\w*|transdifferentiat\w*|lineage\s+plasticity|"
    r"treatment[\s-]+(?:emergent|related|induced)|t-?nepc|"
    r"(?:evolv|progress|convert)\w*\s+(?:in)?to|dedifferentiat\w*)\b",
    flags=re.IGNORECASE,
)
NE_MARKER = re.compile(
    r"\b(?:synaptophysin|chromogranin(?:\s+a)?|cd\s?56|insm-?1|"
    r"neuron[\s-]+specific\s+enolase|nse)\b",
    flags=re.IGNORECASE,
)
# Stains not yet resulted, or not interpretable.
IHC_NOT_RESULTED = re.compile(
    r"\b(?:pending|ordered|requested|sent|will\s+be|to\s+be\s+(?:performed|done)|"
    r"in\s+progress|equivocal|indeterminate|non[\s-]?specific|weak\s+and\s+"
    r"non[\s-]?specific|cannot\s+be\s+(?:assessed|evaluated)|insufficient|"
    r"serum|blood|plasma|ng/ml|ng/l|u/l|iu/l|nmol/l|elevated|level)\b",
    flags=re.IGNORECASE,
)
IHC_STAIN_INTENSITY = re.compile(
    r"\b(?:focal(?:ly)?|diffuse(?:ly)?|weak(?:ly)?|strong(?:ly)?)"
    r"(?:[\s,-]+(?:focal(?:ly)?|diffuse(?:ly)?|weak(?:ly)?|strong(?:ly)?))*\s*$",
    re.IGNORECASE,
)
IHC_STAIN_NOUN = re.compile(r"^\s+(?:staining|immunoreactivity|expression)\b", re.IGNORECASE)

# Negation and hedge cues scanned before each NE term, within one sentence.
# Unlike the met gate this keeps "progression to" OUT (it is how a
# transformation is written) and adds "lack of".
NE_NEGATION_HEDGE = re.compile(
    r"\b(?:no|not|without|negative|neg|absent|absence|lack(?:s|ing)?|lack\s+of|"
    r"free\s+of|rule[\s-]?out|r/o|ruled\s+out|exclude[ds]?|excluding|"
    r"versus|vs|differential|ddx|consider(?:ed|ation)?|"
    r"suspicious|suspect(?:ed)?|suspicion|concern(?:ing|ed)?|"
    r"possible|possibly|probable|probably|likely|unlikely|presumed|"
    r"possibilit(?:y|ies)|question\s+of|look(?:ing)?\s+for|hypothes\w+|doubtful|"
    r"cannot\s+(?:be\s+)?exclude[d]?|equivocal|indeterminate|questionable|"
    r"favor(?:s|ed|ing)?|worrisome|"
    r"if|whether|should|would|may|might|could|"
    r"evaluate|evaluation|assess|work[\s-]?up|"
    r"monitor(?:ing|ed)?|surveillance|watch(?:ing|ed)?\s+for|"
    r"risk\s+(?:of|for)|at\s+risk|potential|pending|await(?:ing|ed)?|"
    r"family\s+history|mother|father|brother|sister)\b",
    flags=re.IGNORECASE,
)
NE_POST_NEGATION = re.compile(
    r"^(?:\W+|[\w/()-]+\s+){0,4}?"
    r"(?:(?:is|was|are|were|has\s+been|have\s+been)\s+(?:not|negative|absent|"
    r"ruled\s+out|excluded|unlikely|less\s+likely)|"
    r"not\s+(?:identified|seen|present|detected|appreciated|demonstrated)|"
    r"negative\b|absent\b|\(\s*-\s*\))",
    flags=re.IGNORECASE,
)
NE_POST_HEDGE = re.compile(
    r"^(?:[\s,:()-]+[\w/-]+){0,4}?[\s,:()-]*"
    r"(?:can(?:not|\s+not)\s+(?:be\s+)?excluded|"
    r"(?:is|was|are|were|remains?)\s+(?:only\s+)?"
    r"(?:suspected|possible|probable|likely|unlikely|presumed|favou?red|"
    r"considered|equivocal|indeterminate|questionable|a\s+possibility)|"
    r"(?:suspected|possible|probable|favou?red|considered|"
    r"(?:less\s+|more\s+|most\s+)?(?:un)?likely|doubtful)\b|"
    r"(?:may|might|could)\s+be\b)",
    flags=re.IGNORECASE,
)
# A question mark on the term itself ("?NEPC", "NEPC?", "neuroendocrine
# features? Awaiting stains"). SENTENCE_END treats "?" as a boundary, so the
# trailing form is matched on the raw text rather than the sentence window.
NE_QUESTION_AFTER = re.compile(r"(?:[ \t,]+[\w/()-]+){0,3}?[ \t]*\?")
NON_SMALL_CELL = re.compile(r"\bnon[\s-]*$", flags=re.IGNORECASE)
# NE tumors of other primaries that name no other organ word the met gate's
# site list would catch, or that are disqualifying even beside "prostate".
NON_PROSTATE_NE = re.compile(
    r"\b(?:sclc|small[\s-]?cell\s+lung|merkel|carcinoid|"
    r"pulmonary\s+(?:origin|primary|small)|lung\s+(?:origin|primary)|"
    r"urothelial|bladder\s+(?:primary|origin|cancer|carcinoma|tumou?r)|"
    r"pancreatic\s+(?:net|neuroendocrine)|pnet|gep[\s-]?nets?|"
    r"well[\s-]differentiated\s+neuroendocrine\s+tumou?r)\b",
    flags=re.IGNORECASE,
)

# Histologic differentiation is distinct from Gleason grade. In particular,
# Gleason 4+5 adenocarcinoma with focal NE staining belongs to group 3.
PD_CARCINOMA = re.compile(
    r"\b(?:poorly[\s-]+differentiated|de[\s-]?differentiated)"
    r"(?:[\s,]+(?!neuro|small|large|oat)[\w/()-]+){0,3}?[\s,]+(?:adeno)?carcinoma\b|"
    r"\b(?:adeno)?carcinoma\s*[,:(-]\s*"
    r"(?:poorly[\s-]+differentiated|de[\s-]?differentiated)\b",
    flags=re.IGNORECASE,
)
ADENO_TERM = re.compile(r"\badeno[\s-]?ca(?:rcinoma)?\b|\badenocarcinoma\b", flags=re.IGNORECASE)
UNKNOWN_PRIMARY = re.compile(
    r"\b(?:unknown|uncertain|undetermined|unestablished)\s+(?:primary|origin)\b|"
    r"\b(?:primary(?:\s+site)?|origin)\s+(?:is\s+)?"
    r"(?:unknown|uncertain|undetermined|unestablished)\b",
    re.IGNORECASE,
)
# An NE carcinoma written as one component of a mixed tumor.
MIXED_CUE = re.compile(
    r"\b(?:mixed|admixed|combined|collision)\s+"
    r"(?:(?:prostatic|prostate|acinar|ductal)\s+)?"
    r"(?:adeno\w*|small[\s-]?cell|neuro[\s-]?endocrine)\b|"
    r"\badeno\w*\s+with\s+(?:an?\s+)?(?:[\w-]+\s+){0,3}?"
    r"(?:small[\s-]?cell|neuro[\s-]?endocrine)(?:\s+[\w-]+){0,2}?\s+component\b",
    flags=re.IGNORECASE,
)

EVIDENCE_TERMS = {
    "nepc_diagnosis": (NE_DIAGNOSIS, "no_ne_diagnosis_term"),
    "histologic_transformation": (TRANSFORMATION, "no_transformation_term"),
    "ne_differentiation": (NE_TERM, "no_ne_term"),
    "small_cell_histology": (SMALL_CELL_TERM, "no_small_cell_term"),
    "mixed_ne_carcinoma": (NE_DIAGNOSIS, "no_ne_diagnosis_term"),
    "pd_carcinoma_ne_features": (PD_CARCINOMA, "no_pd_carcinoma_term"),
}
EVIDENCE_TYPES = frozenset(EVIDENCE_TERMS) | {
    "positive_ne_ihc", "poorly_differentiated_carcinoma", "prostate_adenocarcinoma",
}



def _context_checks(text, hit):
    """Trial-language and non-prostate-primary checks around one occurrence."""
    # Research-specific cues only: met_veto.BOILERPLATE's "registry", "study
    # of" and bare "protocol"/"consent" also occur in real pathology and
    # clinic text ("per CAP protocol", "consented to biopsy").
    trial_reason = trial_boilerplate_reason(text)
    if trial_reason is not None:
        return trial_reason
    if PROSTATE_CUE.search(text) is None:
        site = _is_non_prostate_primary(
            text[max(0, hit.start() - SITE_WINDOW_CHARS):hit.end() + SITE_WINDOW_CHARS]
        )
        if site is not None:
            return f"non_prostate_primary:{site}"
    return None


def _screen_hit(text, hit, term=NE_TERM, context=True):
    """Reason one `term` occurrence is not asserted, or None if it is.

    `context=False` skips the trial and non-prostate-primary checks, for a cue
    word whose site is screened separately.
    """
    if term is NE_TERM and NON_SMALL_CELL.search(text[max(0, hit.start() - 5):hit.start()]):
        return "non_small_cell"
    if hit.start() > 0 and text[hit.start() - 1] == "?":
        return "negated_or_hedged:question"
    before = _same_sentence_before(text, hit.start(), NEGATION_WINDOW_CHARS)
    cue = NE_NEGATION_HEDGE.search(before)
    if cue is not None:
        return "negated_or_hedged:" + " ".join(cue.group(0).lower().split())
    if NE_QUESTION_AFTER.match(text, hit.end()):
        return "negated_or_hedged:question"
    after = _same_sentence_after(text, hit.end(), POST_WINDOW_CHARS)
    if NE_POST_NEGATION.match(after):
        return "negated_or_hedged:post_negation"
    if NE_POST_HEDGE.match(after):
        return "negated_or_hedged:post_hedge"
    return _context_checks(text, hit) if context else None


def _screen_terms(text, term=NE_TERM):
    """Every `term` occurrence must be asserted, not negated or hedged."""
    for hit in term.finditer(text):
        reason = _screen_hit(text, hit, term)
        if reason:
            return reason
    return None


def screen_asserted(quote, term):
    """Return ``(ok, reason)``: is every occurrence of `term` asserted as present?

    The NE gate's sentence-bounded negation/hedge, post-negation, boilerplate,
    and non-prostate-primary checks applied to another term. The AVPC rollup
    uses it for findings the metastatic gate does not cover.
    """
    text = _flatten(quote)
    if term.search(text) is None:
        return False, "no_term"
    reason = _screen_terms(text, term)
    return (False, reason) if reason else (True, None)


def screen_any_asserted(quote, term, context=True):
    """Return ``(ok, reason)``: is at least one occurrence of `term` asserted?

    For a cue word that may legitimately recur negated elsewhere in the quote
    ("no hepatic metastases; new lung metastases"); the first failure reason
    is returned when none survives.
    """
    text = _flatten(quote)
    first = None
    for hit in term.finditer(text):
        reason = _screen_hit(text, hit, term, context)
        if reason is None:
            return True, None
        first = first or reason
    return False, first or "no_term"


def positive_ihc_markers(quote):
    """Return ``(markers, reason)``: NE markers read as positive tissue IHC.

    `markers` is a sorted list of lower-cased marker names; when it is empty,
    `reason` says why the first marker failed.
    """
    text = _flatten(quote)
    found = list(NE_MARKER.finditer(text))
    if not found:
        return [], "no_ne_marker"
    other = NON_PROSTATE_NE.search(text)
    if other is not None:
        return [], f"non_prostate_primary:{other.group(0).lower()}"
    positive, reason = set(), None
    for marker in found:
        polarity = _nearest_polarity(text, marker)
        # Explicit intensity + marker + staining is a reported tissue result,
        # e.g. "focal weak synaptophysin staining", even without "positive".
        intensity = IHC_STAIN_INTENSITY.search(text, 0, marker.start())
        if (polarity is None and intensity
                and IHC_STAIN_NOUN.match(text[marker.end():])):
            polarity = "negative" if _positive_cue_is_negated(text, marker, intensity) else "positive"
        if polarity != "positive":
            reason = reason or ("negative_ihc" if polarity == "negative" else "no_positive_ihc_result")
            continue
        window = text[
            max(0, marker.start() - IHC_POLARITY_PRE_CHARS):marker.end() + IHC_POLARITY_POST_CHARS
        ]
        pending = IHC_NOT_RESULTED.search(window)
        if pending is not None:
            reason = reason or f"ihc_not_resulted:{pending.group(0).lower()}"
            continue
        hedge = NE_NEGATION_HEDGE.search(
            _same_sentence_before(text, marker.start(), IHC_POLARITY_PRE_CHARS)
        )
        if hedge is not None and hedge.group(0).lower() not in {"negative", "neg", "no", "not"}:
            reason = reason or f"negated_or_hedged:{hedge.group(0).lower()}"
            continue
        context = _context_checks(text, marker)
        if context:
            reason = reason or context
            continue
        positive.add(re.sub(r"\s+", " ", marker.group(0).lower()))
    return sorted(positive), (None if positive else reason)


def screen_positive_ihc(quote):
    """Return ``(ok, reason)``: does at least one NE marker read as positive tissue IHC?"""
    markers, reason = positive_ihc_markers(quote)
    return (True, None) if markers else (False, reason)


def screen_pd_carcinoma(quote):
    """Return ``(ok, reason)``: is a poorly differentiated carcinoma asserted?

    No NE term is required: this finding only supports a tier-2 label when the
    same note also has gated positive NE IHC (tasks/nepc/tiers.py).
    """
    text = _flatten(quote)
    if UNKNOWN_PRIMARY.search(text):
        return False, "unestablished_primary"
    if PD_CARCINOMA.search(text) is None:
        return False, "no_pd_carcinoma_term"
    reason = _screen_terms(text, PD_CARCINOMA)
    if reason:
        return False, reason
    other = NON_PROSTATE_NE.search(text)
    if other is not None:
        return False, f"non_prostate_primary:{other.group(0).lower()}"
    return True, None


def screen_ne_quote(quote, evidence_type):
    """Return ``(ok, reason)`` for one grounded neuroendocrine quote.

    Reasons: ``invalid_evidence_type:<t>`` / ``no_ne_diagnosis_term`` /
    ``no_transformation_term`` / ``no_ne_term`` / ``no_small_cell_term`` /
    ``no_ne_marker`` / ``negative_ihc`` / ``no_positive_ihc_result`` /
    ``ihc_not_resulted:<cue>`` / ``non_small_cell`` /
    ``negated_or_hedged:<cue>`` / ``non_prostate_primary:<cue>`` /
    ``boilerplate:<cue>``.
    """
    if evidence_type not in EVIDENCE_TYPES:
        return False, f"invalid_evidence_type:{evidence_type}"
    if evidence_type == "positive_ne_ihc":
        return screen_positive_ihc(quote)
    if evidence_type == "poorly_differentiated_carcinoma":
        return screen_pd_carcinoma(quote)
    text = _flatten(quote)
    if evidence_type in {"prostate_adenocarcinoma", "pd_carcinoma_ne_features"} and UNKNOWN_PRIMARY.search(text):
        return False, "unestablished_primary"
    if evidence_type in {"prostate_adenocarcinoma", "mixed_ne_carcinoma"}:
        if ADENO_TERM.search(text) is None:
            return False, "no_adeno_term"
        reason = _screen_terms(text, ADENO_TERM)
        if reason:
            return False, reason
        if evidence_type == "prostate_adenocarcinoma":
            other = NON_PROSTATE_NE.search(text)
            return (False, f"non_prostate_primary:{other.group(0).lower()}") if other else (True, None)
    pattern, missing_reason = EVIDENCE_TERMS[evidence_type]
    if pattern.search(text) is None:
        return False, missing_reason
    # A transformation must still name what it transformed to.
    if NE_TERM.search(text) is None:
        # A PD carcinoma with explicitly positive NE tissue markers also
        # supports group 2 without spelling out "neuroendocrine features".
        if evidence_type != "pd_carcinoma_ne_features":
            return False, "no_ne_term"
        ok, reason = screen_positive_ihc(text)
        if not ok:
            return False, reason
    else:
        reason = _screen_terms(text)
    if reason:
        return False, reason
    if evidence_type == "histologic_transformation":
        reason = _screen_terms(text, TRANSFORMATION)
        if reason:
            return False, reason
    if evidence_type == "pd_carcinoma_ne_features":
        reason = _screen_terms(text, PD_CARCINOMA)
        if reason:
            return False, reason
    other = NON_PROSTATE_NE.search(text)
    if other is not None:
        return False, f"non_prostate_primary:{other.group(0).lower()}"
    return True, None
