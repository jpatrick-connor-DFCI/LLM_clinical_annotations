"""Deterministic quote gate for AR-low / AR-indifferent findings.

Applied at label-build time to every grounded ARLPC finding; the model cannot
override it. Precision is the objective: an ambiguous quote is rejected, which
costs one finding rather than the patient's label.

  - IHC is read per marker with the shared polarity engine
    (`tasks/ihc_polarity.py`). Unlike the NE gate, an equidistant
    positive/negative tie is UNRESOLVED, not negative: negative is the target
    here, so the NE gate's conservative tie rule would be anti-conservative.
  - A marker reads "reduced" when its own clause has reduced wording (weak,
    low, focal, patchy, heterogeneous, reduced vs prior...) and no moderate or
    strong wording. Wording decides; percentages and H-scores are stored only.
  - Origin veto: AR-low staining is a classic clue to a non-prostate primary,
    so a quote that favors one (or a model `favored_origin` of non_prostate)
    is rejected even when it also mentions the prostate.
  - Genomic AR loss must read as a somatic sequencing result; germline, AR-V7,
    amplification, gain and mutation are AR-driven or not tumor evidence.

Bump AR_GATE_VERSION on any change. Label metadata records it, and
--rebuild-labels-only re-applies a changed gate without LLM calls.
"""

import re

from tasks.ihc_polarity import (
    CLAUSE_AFTER_CHARS,
    IHC_ASSERTION_NEGATION,
    IHC_NEGATIVE,
    IHC_POLARITY_PRE_CHARS,
    IHC_POST_NEGATION,
    flatten,
    nearest_polarity,
    marker_context as _marker_context,
    same_sentence_after,
    same_sentence_before,
)
from tasks.met_veto import (
    NON_PROSTATE_SITE,
    PROSTATE_CUE,
    _is_non_prostate_primary,
)
from tasks.nepc.ne_gate import NE_POST_HEDGE
from tasks.trial_guard import trial_boilerplate_reason

AR_GATE_VERSION = "ar-gate-v3"

AR_MARKER = re.compile(
    r"\b(?:androgen\s+receptors?|ar)\b(?![\s-]*v7)(?!\s+(?:amplif|mutat|gain))",
    flags=re.IGNORECASE,
)
TARGET_MARKERS = {
    "psa": re.compile(r"\b(?:psa|prostate[\s-]+specific\s+antigen)\b(?![\s-]*membrane)", re.IGNORECASE),
    "psap": re.compile(r"\b(?:psap|prostatic\s+acid\s+phosphatase)\b", re.IGNORECASE),
    "nkx3_1": re.compile(r"\bnkx[\s-]?3[.\s-]?1\b", re.IGNORECASE),
    "psma": re.compile(r"\b(?:psma|prostate[\s-]+specific\s+membrane\s+antigen)\b", re.IGNORECASE),
    "hoxb13": re.compile(r"\bhoxb[\s-]?13\b", re.IGNORECASE),
}
AR_REDUCED = re.compile(
    r"\b(?:weak\w*|low|focal\w*|patchy|heterogeneous\w*|reduced|decreased|"
    r"diminished|subset|rare|scattered)\b",
    flags=re.IGNORECASE,
)
AR_STRONG = re.compile(r"\b(?:moderate\w*|strong\w*|intense\w*)\b", flags=re.IGNORECASE)
AR_LOST = re.compile(r"\b(?:lost|loss|null|absent)\b", flags=re.IGNORECASE)
# Stains not resulted or not interpretable. No serum cues: those are handled
# per marker (PSA) so they cannot veto an AR stain beside a serum PSA value.
AR_NOT_RESULTED = re.compile(
    r"\b(?:pending|ordered|requested|will\s+be|to\s+be\s+(?:performed|done|sent)|"
    r"in\s+progress|equivocal|indeterminate|non[\s-]?specific|"
    r"cannot\s+be\s+(?:assessed|evaluated)|insufficient|failed|uninterpretable)\b",
    flags=re.IGNORECASE,
)
# Hedges before a marker. No negative words (negative is the target) and no
# "favor" (a favored origin is handled by the origin veto).
AR_HEDGE = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspect\w*|suspicious|concern\w*|"
    r"consider\w*|rule[\s-]?out|r/o|if|whether|would|may|might|could|"
    r"differential|versus|vs|questionable|equivocal)\b",
    flags=re.IGNORECASE,
)

FAVOR_CUE = (
    r"(?:favou?r(?:s|ed|ing)?|suggestive\s+of|in\s+keeping\s+with|"
    r"consistent\s+with|supports?|compatible\s+with)"
)
FAVOR_NON_PROSTATE = re.compile(
    FAVOR_CUE + r"\W+(?:[\w/-]+\W+){0,4}?(" + NON_PROSTATE_SITE.pattern + r")",
    flags=re.IGNORECASE,
)
AGAINST_PROSTATE = re.compile(
    r"\b(?:rather\s+than\s+(?:a\s+)?prostat\w*|not\s+(?:of\s+)?prostat\w*\s+origin|"
    r"argues?\s+against\s+(?:a\s+)?prostat\w*|non[\s-]?prostatic)",
    flags=re.IGNORECASE,
)

AR_STATEMENT_TERM = re.compile(
    r"\b(?:ar|androgen\s+receptor)[\s-]*(?:null|negative|neg|low|indifferen\w*|"
    r"independen\w*|loss|lost|deficient)\b|\bdouble[\s-]?negative\b|\bdnpc\b|\barlpc\b",
    flags=re.IGNORECASE,
)
DOUBLE_NEGATIVE = re.compile(r"\bdouble[\s-]?negative\b", flags=re.IGNORECASE)
DOUBLE_NEGATIVE_CONTEXT = re.compile(
    r"\b(?:ar|androgen|ne|neuro[\s-]?endocrine|prostat\w*|crpc|mcrpc)\b", flags=re.IGNORECASE
)
ANDRO_INDEP_TERM = re.compile(
    r"\b(?:androgen[\s-]+(?:independen\w*|indifferen\w*)|"
    r"hormone[\s-]+(?:refractory|resistan\w*))\b",
    flags=re.IGNORECASE,
)
# Negation/hedge before a statement term, within one sentence. Deliberately
# lacks "negative": "NE-negative, AR-negative" must not veto itself.
STATEMENT_NEG_HEDGE = re.compile(
    r"\b(?:no|not|never|without|absence\s+of|rule[\s-]?out|r/o|ruled\s+out|"
    r"exclude[ds]?|possible|possibly|probable|probably|likely|unlikely|"
    r"suspect\w*|suspicious|concern\w*|consider\w*|question\w*|if|whether|"
    r"would|may|might|could|risk\s+(?:of|for)|monitor\w*|evaluat\w*|"
    r"versus|vs|differential|family\s+history)\b",
    flags=re.IGNORECASE,
)
STATEMENT_POST_NEGATION = re.compile(
    r"^(?:\W+|[\w/()-]+\s+){0,3}?(?:(?:is|was|are|were)\s+(?:not|unlikely|"
    r"ruled\s+out|excluded)|not\s+(?:identified|seen|present|confirmed))\b",
    flags=re.IGNORECASE,
)
STATEMENT_WINDOW_CHARS = 60
STATEMENT_POST_CHARS = 40

GENOMIC_TERM = re.compile(
    r"\b(?:ar|androgen\s+receptor)\b\s*(?:gene\s+)?(?:homozygous\s+|biallelic\s+|"
    r"copy[\s-]+(?:number\s+)?)?(?:loss|deletion|deleted)\b|"
    r"\b(?:loss|deletion)\s+of\s+(?:the\s+)?(?:ar|androgen\s+receptor)\b",
    flags=re.IGNORECASE,
)
GENOMIC_CONTEXT = re.compile(
    r"\b(?:ngs|sequenc\w*|oncopanel|profile|foundation\w*|guardant\w*|tempus|caris|"
    r"copy[\s-]+number|cnv|deletion|deleted|homozygous|biallelic|ctdna|cfdna|"
    r"liquid\s+biopsy|genomic\w*|somatic|panel)\b",
    flags=re.IGNORECASE,
)
GENOMIC_AR_DRIVEN = re.compile(
    r"\b(?:germline|v7|amplif\w*|mutation|mutated|gain|t878a|l702h|h875y|w742c)\b",
    flags=re.IGNORECASE,
)

PSA_SERUM = re.compile(
    r"\b(?:serum|blood|level|nadir|doubling|velocity|ng/ml|ng/dl|undetectable)\b|"
    r"\bpsa\s*(?:of|was|is|=|:)?\s*<?\s*\d",
    flags=re.IGNORECASE,
)
TISSUE_CUE = re.compile(r"\b(?:stain\w*|immuno\w*|ihc|tumou?r\s+cells)\b", flags=re.IGNORECASE)
PSMA_NOT_IHC = re.compile(
    r"\b(?:pet|scan|suv\w*|avid\w*|uptake|pylarify|dcfpyl|ga[\s-]?68|pluvicto|"
    r"lutetium|lu[\s-]?177|radioligand)\b",
    flags=re.IGNORECASE,
)

LOW_READINGS = {"negative", "reduced"}



def marker_reading(text, marker):
    """"negative" / "reduced" / "positive" / None for one marker occurrence."""
    clause, local_marker = _marker_context(text, marker)
    # A negated loss/reduction is not evidence of a low result. Check before
    # generic polarity: "expression" itself is otherwise a positive cue that
    # the shared reader flips in "AR expression is not reduced".
    for pattern in (AR_LOST, AR_REDUCED, IHC_NEGATIVE):
        for cue in pattern.finditer(clause):
            if (IHC_ASSERTION_NEGATION.search(clause[:cue.start()])
                    or IHC_POST_NEGATION.match(clause[max(cue.end(), local_marker.end()):])):
                return None
    base = nearest_polarity(clause, local_marker, tie=None)
    if base == "negative" or AR_LOST.search(clause):
        return "negative"
    if AR_REDUCED.search(clause) and not AR_STRONG.search(clause):
        return "reduced"
    if base == "positive" or AR_STRONG.search(clause):
        return "positive"
    return None


def _marker_checks(text, marker):
    """Not-resulted and hedge checks around one marker occurrence."""
    window = same_sentence_before(text, marker.start(), IHC_POLARITY_PRE_CHARS)
    after = same_sentence_after(text, marker.end(), CLAUSE_AFTER_CHARS)
    pending = AR_NOT_RESULTED.search(window + " " + after)
    if pending is not None:
        return f"ihc_not_resulted:{pending.group(0).lower()}"
    hedge = AR_HEDGE.search(window)
    if hedge is not None:
        return f"negated_or_hedged:{hedge.group(0).lower()}"
    return None


def origin_veto(text, favored_origin=None):
    """Reason string when the quote (or model) favors a non-prostate origin."""
    if favored_origin == "non_prostate":
        return "non_prostate_origin:model"
    for favor in FAVOR_NON_PROSTATE.finditer(text):
        after = same_sentence_after(text, favor.end(1), 160)
        before = text[favor.start():favor.start(1)]
        # A site explicitly described as a prostate metastasis is a
        # destination, not a competing primary. A prostate history elsewhere
        # in the quote must not excuse a favored lung/urothelial primary.
        destination = re.match(
            r"\s+metastas(?:is|es)\s+(?:from|of)\s+(?:the\s+)?prostat\w*\b",
            after, re.IGNORECASE,
        ) or re.search(
            r"\bmetastatic\s+prostat\w*\s+(?:\w+\s+){0,3}"
            r"(?:to|in|involving)\s+(?:the\s+)?$", before, re.IGNORECASE,
        )
        if not destination:
            return f"non_prostate_origin:{favor.group(1).lower()}"
    against = AGAINST_PROSTATE.search(text)
    if against is not None:
        return "non_prostate_origin:against_prostate"
    if PROSTATE_CUE.search(text) is None:
        site = _is_non_prostate_primary(text)
        if site is not None:
            return f"non_prostate_primary:{site}"
    return None


def _read_markers(text, pattern, extra_check=None):
    """Readings for every occurrence that passes the checks; first failure reason."""
    readings, reason = [], None
    for marker in pattern.finditer(text):
        failure = _marker_checks(text, marker) or (extra_check and extra_check(text, marker))
        if failure:
            reason = reason or failure
            continue
        reading = marker_reading(text, marker)
        if reading is None:
            reason = reason or "no_ihc_result"
            continue
        readings.append(reading)
    return readings, reason


def screen_ar_ihc(quote, ar_result, favored_origin=None):
    """Return ``(ok, reason, ar_low)`` for an AR IHC quote.

    `ar_low` is True for a negative/reduced reading, False for an AR-positive
    (audit-only) reading. The model's `ar_result` must agree with the quote.
    """
    text = flatten(quote)
    if AR_MARKER.search(text) is None:
        return False, "no_ar_marker", None
    if ar_result is None:
        return False, "missing_ar_result", None
    model_low = ar_result in LOW_READINGS
    if model_low:
        veto = origin_veto(text, favored_origin)
        if veto:
            return False, veto, None
    readings, reason = _read_markers(text, AR_MARKER)
    if model_low and LOW_READINGS & set(readings):
        return True, None, True
    if not model_low and "positive" in readings:
        return True, None, False
    return False, (reason if not readings and reason else "ar_result_not_in_quote"), None


def _target_extra_check(name):
    def check(text, marker):
        sentence = (same_sentence_before(text, marker.start(), 120) + marker.group(0)
                    + same_sentence_after(text, marker.end(), 60))
        if name == "psma" and PSMA_NOT_IHC.search(sentence):
            return "psma_not_ihc"
        if name == "psa":
            if PSA_SERUM.search(sentence):
                return "psa_serum"
        return None
    return check


def ar_target_loss(quote, marker=None, favored_origin=None, note_type=None):
    """Return ``(markers, reason)``: AR-target markers read negative/reduced on tissue."""
    text = flatten(quote)
    veto = origin_veto(text, favored_origin)
    if veto:
        return [], veto
    low, reason, found = set(), None, False
    for name, pattern in TARGET_MARKERS.items():
        if pattern.search(text) is None:
            continue
        found = True
        if name == "psa" and note_type != "Pathology" and TISSUE_CUE.search(text) is None:
            reason = reason or "psa_not_tissue"
            continue
        readings, failure = _read_markers(text, pattern, _target_extra_check(name))
        if LOW_READINGS & set(readings):
            low.add(name)
        else:
            reason = reason or failure or "no_low_target_result"
    if not found:
        return [], "no_ar_target_marker"
    if marker is not None and marker not in low:
        return [], reason or "marker_not_low_in_quote"
    return sorted(low), (None if low else reason)


def _screen_statement(text, term):
    hits = list(term.finditer(text))
    if not hits:
        return "no_term"
    for hit in hits:
        if DOUBLE_NEGATIVE.fullmatch(hit.group(0)):
            sentence = (same_sentence_before(text, hit.start(), 120)
                        + same_sentence_after(text, hit.end(), 120))
            if DOUBLE_NEGATIVE_CONTEXT.search(sentence) is None:
                return "double_negative_without_context"
        before = same_sentence_before(text, hit.start(), STATEMENT_WINDOW_CHARS)
        cue = STATEMENT_NEG_HEDGE.search(before)
        if cue is not None:
            return f"negated_or_hedged:{cue.group(0).lower()}"
        after = same_sentence_after(text, hit.end(), STATEMENT_POST_CHARS)
        if STATEMENT_POST_NEGATION.match(after):
            return "negated_or_hedged:post_negation"
        if NE_POST_HEDGE.match(after):
            return "negated_or_hedged:post_hedge"
    reason = trial_boilerplate_reason(text)
    if reason:
        return reason
    return origin_veto(text)


def screen_statement(quote, evidence_type):
    term = AR_STATEMENT_TERM if evidence_type == "ar_statement" else ANDRO_INDEP_TERM
    reason = _screen_statement(flatten(quote), term)
    return (False, reason) if reason else (True, None)


def screen_genomic(quote):
    text = flatten(quote)
    hits = list(GENOMIC_TERM.finditer(text))
    if not hits:
        return False, "no_genomic_ar_loss_term"
    if GENOMIC_CONTEXT.search(text) is None:
        return False, "no_genomic_context"
    for hit in hits:
        sentence = (same_sentence_before(text, hit.start(), 200) + hit.group(0)
                    + same_sentence_after(text, hit.end(), 200))
        driven = GENOMIC_AR_DRIVEN.search(sentence)
        if driven is not None:
            return False, f"genomic_ar_driven:{driven.group(0).lower()}"
    reason = _screen_statement(text, GENOMIC_TERM)
    return (False, reason) if reason else (True, None)


def screen_ar_finding(row):
    """Return ``(ok, reason, info)`` for one grounded ARLPC finding row.

    info: ``{"ar_low": bool}`` for ar_ihc, ``{"markers": [...]}`` for
    ar_target_ihc, else ``{}``.
    """
    evidence_type = row.get("evidence_type")
    quote = row.get("quote")
    if evidence_type == "ar_ihc":
        ok, reason, low = screen_ar_ihc(quote, row.get("ar_result"), row.get("favored_origin"))
        return ok, reason, {"ar_low": low}
    if evidence_type == "ar_target_ihc":
        markers, reason = ar_target_loss(
            quote, row.get("marker"), row.get("favored_origin"), row.get("note_type")
        )
        return bool(markers), reason, {"markers": markers}
    if evidence_type in {"ar_statement", "androgen_independent_statement"}:
        ok, reason = screen_statement(quote, evidence_type)
        return ok, reason, {}
    if evidence_type == "ar_genomic_loss":
        ok, reason = screen_genomic(quote)
        return ok, reason, {}
    return False, f"invalid_evidence_type:{evidence_type}", {}
