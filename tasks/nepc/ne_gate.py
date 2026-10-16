"""Deterministic quote gate for neuroendocrine findings.

Applied at label-build time to every grounded NEPC finding, and reused by the
AVPC rollup for C1 (small-cell histology) and C6's tissue marker. The model
cannot override it. Precision is the objective: an ambiguous quote is
rejected, which costs one finding rather than the patient's label.

It follows `tasks/longitudinal_subtype/met_veto.py`'s conventions: flatten,
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

from tasks.longitudinal_subtype.met_veto import (
    NEGATION_WINDOW_CHARS,
    POST_WINDOW_CHARS,
    PROSTATE_CUE,
    SITE_WINDOW_CHARS,
    _is_non_prostate_primary,
)
from tasks.trial_guard import trial_boilerplate_reason

NE_GATE_VERSION = "ne-gate-v3"

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
IHC_POSITIVE = re.compile(
    r"(?:\b(?:positive|positivity|pos|reactive|reactivity|immunoreactiv\w*|"
    r"express\w*|stain(?:s|ed|ing)?\s+(?:positive|strongly|diffusely|focally)|"
    r"label(?:s|ed|ing)?|highlight\w*|retained)\b|\(\s*\+\s*\)|(?<=\w)\+|\s\+(?=\s|,|;|$))",
    flags=re.IGNORECASE,
)
IHC_NEGATIVE = re.compile(
    r"(?:\b(?:negative|negativity|neg|nonreactive|non-reactive|unreactive|"
    r"no\s+(?:staining|expression|reactivity)|not\s+(?:expressed|reactive|"
    r"stained|staining|positive)|lack\w*|absent|loss\s+of)\b|\(\s*-\s*\)|"
    r"(?<=\w)-(?=\s|,|;|$))",
    flags=re.IGNORECASE,
)
_BINDS_FORWARD = re.compile(r"\s+for\b", flags=re.IGNORECASE)
IHC_ASSERTION_NEGATION = re.compile(
    r"\b(?:no|not|never|without|absence|lack\w*|neither)\b", flags=re.IGNORECASE
)
IHC_CLAUSE_BREAK = re.compile(r",|\b(?:but|whereas|however)\b", flags=re.IGNORECASE)
IHC_RESULT_NOUN = re.compile(r"\b(?:staining|expression|reactivity)\b", flags=re.IGNORECASE)
IHC_POST_NEGATION = re.compile(
    r"^\s*(?:(?:staining|expression|reactivity)\s+)?"
    r"(?:(?:is|was|are|were|has\s+been|have\s+been)\s+)?"
    r"(?:(?:not|never)\s+(?:identified|seen|present|detected|observed|"
    r"appreciated|demonstrated|expressed)|absent|negative)\b",
    flags=re.IGNORECASE,
)
IHC_POLARITY_PRE_CHARS = 60
IHC_POLARITY_POST_CHARS = 30
# Stains not yet resulted, or not interpretable.
IHC_NOT_RESULTED = re.compile(
    r"\b(?:pending|ordered|requested|sent|will\s+be|to\s+be\s+(?:performed|done)|"
    r"in\s+progress|equivocal|indeterminate|non[\s-]?specific|weak\s+and\s+"
    r"non[\s-]?specific|cannot\s+be\s+(?:assessed|evaluated)|insufficient|"
    r"serum|blood|plasma|ng/ml|ng/l|u/l|iu/l|nmol/l|elevated|level)\b",
    flags=re.IGNORECASE,
)

# Negation and hedge cues scanned before each NE term, within one sentence.
# Unlike the met gate this keeps "progression to" OUT (it is how a
# transformation is written) and adds "lack of".
NE_NEGATION_HEDGE = re.compile(
    r"\b(?:no|not|without|negative|neg|absent|absence|lack(?:s|ing)?|lack\s+of|"
    r"free\s+of|rule[\s-]?out|r/o|ruled\s+out|exclude[ds]?|excluding|"
    r"versus|vs|differential|ddx|consider(?:ed|ation)?|"
    r"suspicious|suspect(?:ed)?|suspicion|concern(?:ing|ed)?|"
    r"possible|possibly|probable|probably|likely|unlikely|presumed|"
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
    r"(?:suspected|possible|probable|favou?red|considered)\b|"
    r"(?:may|might|could)\s+be\b)",
    flags=re.IGNORECASE,
)
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
_SENTENCE_END = re.compile(r"(?<!\bvs)(?<!\bdr)(?<!\bno)[.;!?](?=\s|$)", flags=re.IGNORECASE)

EVIDENCE_TERMS = {
    "nepc_diagnosis": (NE_DIAGNOSIS, "no_ne_diagnosis_term"),
    "histologic_transformation": (TRANSFORMATION, "no_transformation_term"),
    "ne_differentiation": (NE_TERM, "no_ne_term"),
    "small_cell_histology": (SMALL_CELL_TERM, "no_small_cell_term"),
}
EVIDENCE_TYPES = frozenset(EVIDENCE_TERMS) | {"positive_ne_ihc"}


def _flatten(quote):
    return re.sub(r"\s+", " ", str(quote or "")).strip()


def _same_sentence_before(text, start, window):
    before = text[max(0, start - window):start]
    ends = list(_SENTENCE_END.finditer(before))
    return before[ends[-1].end():] if ends else before


def _same_sentence_after(text, end, window):
    after = text[end:end + window]
    boundary = _SENTENCE_END.search(after)
    return after[:boundary.start()] if boundary else after


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


def _screen_terms(text, term=NE_TERM):
    """Every `term` occurrence must be asserted, not negated or hedged."""
    for hit in term.finditer(text):
        if term is NE_TERM and NON_SMALL_CELL.search(text[max(0, hit.start() - 5):hit.start()]):
            return "non_small_cell"
        before = _same_sentence_before(text, hit.start(), NEGATION_WINDOW_CHARS)
        cue = NE_NEGATION_HEDGE.search(before)
        if cue is not None:
            return f"negated_or_hedged:{cue.group(0).lower()}"
        after = _same_sentence_after(text, hit.end(), POST_WINDOW_CHARS)
        if NE_POST_NEGATION.match(after):
            return "negated_or_hedged:post_negation"
        if NE_POST_HEDGE.match(after):
            return "negated_or_hedged:post_hedge"
        reason = _context_checks(text, hit)
        if reason:
            return reason
    return None


def _positive_cue_is_negated(text, marker, cue):
    """Read negation on this marker's result, without borrowing other panel results."""
    before = _same_sentence_before(
        text, min(marker.start(), cue.start()), IHC_POLARITY_PRE_CHARS
    )
    before = IHC_CLAUSE_BREAK.split(before)[-1]
    # A conjunction after a completed result starts another result clause:
    # "no synaptophysin positivity and chromogranin positive". A marker list
    # such as "no synaptophysin and chromogranin positivity" keeps its negation.
    for conjunction in reversed(list(re.finditer(r"\b(?:and|or)\b", before, re.IGNORECASE))):
        preceding = before[:conjunction.start()]
        if (IHC_POSITIVE.search(preceding) or IHC_NEGATIVE.search(preceding)
                or IHC_RESULT_NOUN.search(preceding)):
            before = before[conjunction.end():]
            break
    if IHC_ASSERTION_NEGATION.search(before):
        return True
    after = _same_sentence_after(
        text, max(marker.end(), cue.end()), IHC_POLARITY_POST_CHARS
    )
    return IHC_POST_NEGATION.match(after) is not None


def _nearest_polarity(text, marker):
    """Return "positive" / "negative" / None for the cue nearest one marker.

    A cue written "<polarity> for" binds forward ("positive for synaptophysin
    and negative for chromogranin"), so it only counts for markers after it.
    Ties go to negative, the conservative reading.
    """
    lo = max(0, marker.start() - IHC_POLARITY_PRE_CHARS)
    hi = marker.end() + IHC_POLARITY_POST_CHARS
    best = None
    for polarity, pattern in (("negative", IHC_NEGATIVE), ("positive", IHC_POSITIVE)):
        for cue in pattern.finditer(text, lo, min(hi, len(text))):
            if cue.end() <= marker.start():
                distance = marker.start() - cue.end()
            elif cue.start() >= marker.end():
                if _BINDS_FORWARD.match(text, cue.end()):
                    continue
                distance = cue.start() - marker.end()
            else:
                continue
            result = polarity
            if polarity == "positive" and _positive_cue_is_negated(text, marker, cue):
                result = "negative"
            if (best is None or distance < best[0]
                    or (distance == best[0] and result == "negative")):
                best = (distance, result)
    return best[1] if best else None


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
    text = _flatten(quote)
    pattern, missing_reason = EVIDENCE_TERMS[evidence_type]
    if pattern.search(text) is None:
        return False, missing_reason
    # A transformation must still name what it transformed to.
    if NE_TERM.search(text) is None:
        return False, "no_ne_term"
    reason = _screen_terms(text)
    if reason:
        return False, reason
    if evidence_type == "histologic_transformation":
        reason = _screen_terms(text, TRANSFORMATION)
        if reason:
            return False, reason
    other = NON_PROSTATE_NE.search(text)
    if other is not None:
        return False, f"non_prostate_primary:{other.group(0).lower()}"
    return True, None
