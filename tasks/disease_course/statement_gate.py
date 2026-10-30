"""Label-time screens for disease-course findings.

The extraction runner stores every grounded finding; these screens decide which
ones the course labels and the BCR, progression and mCRPC builders may use.
They run at label time, so `--rebuild-labels-only` and every builder re-apply
a change here without LLM calls.

Every finding first passes the source-context trial screen
(`_trial_context_reason`, attached by tasks.note_extraction from the evidence
and including the quote-level trial check). Then, by item:

  psa_value                no threshold, goal or hypothetical framing around
                           the number
  radical_prostatectomy,   the procedure or agent is named and was done, not
  radiation_therapy,       planned, recommended, declined, or negated
  therapy_start
  therapy_current          named, not negated or planned
  therapy_stop             named with a stop cue, not planned
  bcr_statement,           asserted (tasks/nepc/ne_gate.screen_asserted, which
  progression_event        is sentence-bounded) and not time-to-event,
                           progression-free or conditional language
  crpc_statement           mcrpc_gate.screen_castration_resistant_quote
  mcrpc_statement          an mCRPC term, not nmCRPC, passing both the CRPC and
                           the metastatic screen (so N1-only does not count)
  imaging_recurrence       certainty-aware: definite and probable ("suspicious
                           for") findings pass; negated, equivocal, hypothetical
                           and planned ones do not. A hedge in the quote caps
                           the model's certainty at "probable".

No existing gate is changed. Bump COURSE_GATE_VERSION on any change; label
metadata records it.
"""

import re

from tasks.longitudinal_subtype.mcrpc_gate import (
    _CASTRATING_THERAPY,
    screen_castration_resistant_quote,
    screen_metastatic_quote,
)
from tasks.nepc.ne_gate import screen_asserted

COURSE_GATE_VERSION = "course-gate-v1"

SENTENCE_WINDOW_CHARS = 120
NOT_GIVEN_WORDS = 3
_SENTENCE_END = re.compile(r"(?<!\bvs)(?<!\bdr)(?<!\bno)[.;!?](?=\s|$)", flags=re.IGNORECASE)
_WORDS_BEFORE = re.compile(rf"(?:[\w/'-]+\W*){{0,{NOT_GIVEN_WORDS}}}$")

# Cues that a procedure or drug was not (yet) given, scanned in the same
# sentence before the term.
NOT_GIVEN_BEFORE = re.compile(
    r"\b(?:no|not|never|without|declin\w*|refus\w*|defer\w*|decid\w*\s+against|"
    r"candidate|consider\w*|recommend\w*|plan\w*|discuss\w*|offer\w*|option\w*|"
    r"propos\w*|will|would|could|may|might|should|if|whether|"
    r"schedul\w*|pending|referr?\w*|await\w*|eligible|counsel\w*|"
    r"prior\s+to\s+starting|before\s+starting|to\s+start|"
    r"family\s+history|father|brother)\b",
    flags=re.IGNORECASE,
)
# ... and right after it ("RALP was recommended", "Lupron to be started").
NOT_GIVEN_AFTER = re.compile(
    r"^[\s,:()-]*(?:(?:is|was|were|has\s+been|will\s+be|would\s+be)\s+)?"
    r"(?:recommend\w*|plann\w*|discuss\w*|offer\w*|consider\w*|declin\w*|"
    r"defer\w*|schedul\w*|pending|to\s+be\s+\w+|not\s+\w+|deferred)\b",
    flags=re.IGNORECASE,
)

PROSTATECTOMY_TERM = re.compile(
    r"\b(?:(?:radical\s+)?(?:retropubic\s+|perineal\s+|robotic\s+|laparoscopic\s+)?"
    r"prostatectomy|ralp|rarp|rrp|lrp|rp)\b",
    flags=re.IGNORECASE,
)
RADIATION_TERM = re.compile(
    r"\b(?:radiation|radiotherapy|irradiat\w*|xrt|imrt|ebrt|sbrt|sabr|igrt|"
    r"brachytherapy|brachy|proton|seed\s+implant\w*|seeds|rt|cyberknife|gy)\b",
    flags=re.IGNORECASE,
)
SYSTEMIC_TERM = re.compile(
    rf"\b(?:{_CASTRATING_THERAPY}|bicalutamide|casodex|flutamide|nilutamide|"
    r"histrelin|vantas|supprelin|camcevi|docetaxel|taxotere|cabazitaxel|jevtana|"
    r"chemo(?:therapy)?|radium(?:[\s-]*223)?|ra[\s-]*223|xofigo|"
    r"lutetium|lu[\s-]*177|177[\s-]*lu|pluvicto|vipivotide|psma[\s-]*617|"
    r"sipuleucel(?:[\s-]*t)?|provenge|olaparib|lynparza|rucaparib|rubraca|"
    r"niraparib|zejula|akeega|talazoparib|talzenna|parp\s+inhibitors?|"
    r"carboplatin|cisplatin|platinum)\b",
    flags=re.IGNORECASE,
)
STOP_CUE = re.compile(
    r"\b(?:stop\w*|discontinu\w*|d/c|held|hold\w*|complet\w*|finish\w*|ceased|"
    r"off|switch\w*\s+from|transitioned\s+off|last\s+(?:dose|injection|cycle)|"
    r"holiday|intermittent|withdr[ae]w\w*)\b",
    flags=re.IGNORECASE,
)
STOP_PLANNED = re.compile(
    r"\b(?:plan\w*|recommend\w*|consider\w*|discuss\w*|will|would|if|should|"
    r"may|might|could|option\w*)\b(?:\W+\w+){0,4}?\W+(?:stop|discontinu|hold|"
    r"complet|finish|switch)\w*",
    flags=re.IGNORECASE,
)

BCR_TERM = re.compile(
    r"\b(?:biochemical(?:ly)?\s+(?:recurren\w*|relapse\w*|failure|progression|"
    r"persisten\w*)|bcr|bcf|"
    r"psa\s+(?:recurren\w*|relapse\w*|failure|progression|persistence)|"
    r"(?:recurren\w*|relapse\w*)|persistent(?:ly)?\s+(?:elevated\s+|detectable\s+)?psa|"
    r"detectable\s+psa|phoenix)\b",
    flags=re.IGNORECASE,
)
PROGRESSION_TERM = re.compile(
    r"\b(?:progress(?:ion|ed|ing|ive)|new\s+(?:\w+\s+){0,3}?(?:lesions?|metastas\w+|"
    r"mets|nodes?|foci|disease|sites?)|"
    r"(?:interval\s+)?(?:increas\w*|enlarg\w*|grow\w*)\s+(?:in\s+(?:size|number|extent)|of)|"
    r"(?:increas\w*|enlarg\w*|growing|worsen\w*)|rising\s+psa|"
    r"psa\s+(?:rise|rose|rising|increas\w*|doubl\w*))\b",
    flags=re.IGNORECASE,
)
# Time-to-event, definitional and conditional language: never a dated event.
NOT_AN_EVENT = re.compile(
    r"\b(?:(?:progression|recurrence|relapse|bcr|metastas[ie]s|disease|event)"
    r"[\s-]+free|r?pfs|mfs|"
    r"time\s+to\s+(?:\w+\s+){0,2}?(?:progression|recurrence|relapse|failure)|"
    r"until\s+(?:\w+\s+){0,2}?(?:progression|recurrence|relapse|intolerance)|"
    r"(?:upon|at\s+(?:the\s+time\s+of\s+)?|on|after|before|prior\s+to)\s+"
    r"(?:further\s+|future\s+|radiographic\s+|psa\s+|biochemical\s+)?progression|"
    r"risk\s+(?:of|for)\s+(?:\w+\s+){0,2}?(?:progression|recurrence|relapse|failure)|"
    r"(?:if|when|once|should)\s+(?:\w+\s+){0,4}?(?:progress\w*|recur\w*|rises?|rising)|"
    r"defined\s+as|definition|criteria|"
    r"(?:in\s+the\s+)?absence\s+of\s+(?:\w+\s+){0,2}?(?:progression|recurrence))\b",
    flags=re.IGNORECASE,
)
STABLE = re.compile(
    r"\b(?:stable|unchanged|no\s+(?:significant\s+)?(?:interval\s+)?change|"
    r"similar|grossly\s+stable|without\s+(?:interval\s+)?change)\b",
    flags=re.IGNORECASE,
)
PSA_HYPOTHETICAL = re.compile(
    r"\b(?:if|unless|once|when|goal|target|threshold|cut[\s-]?off|nadir\s*\+|"
    r"defined\s+as|criteri\w*|greater\s+than|above|exceeds?|reaches?|"
    r"should|would|expect\w*)\b",
    flags=re.IGNORECASE,
)

MCRPC_TERM = re.compile(
    r"\b(?:mcrpc|metastatic\s+(?:castrat(?:e|ion)|hormone|androgen)[\s-]+"
    r"(?:resistan\w*|refractory|independen\w*|insensitive)|metastatic\s+hrpc)\b",
    flags=re.IGNORECASE,
)
NON_METASTATIC = re.compile(r"\b(?:nm[\s-]?crpc|non[\s-]?metastatic|m0)\b", flags=re.IGNORECASE)

# Imaging certainty. The quote decides; the model's enum can only be lowered.
IMAGING_NEGATED = re.compile(
    r"\b(?:no\s+(?:\w+\s+){0,2}?(?:evidence|definite|definitive|suspicious|"
    r"pet[\s-]avid|psma[\s-]avid|avid|focal|abnormal\w*|recurren\w*|metasta\w*|"
    r"lesions?|uptake|disease)|negative\s+(?:for|study|scan|exam\w*|psma|pet)|"
    r"without\s+(?:evidence|definite|definitive)|not\s+(?:seen|identified|"
    r"demonstrated|visualized|appreciated)|resolved|resolution|unremarkable|"
    r"complete\s+(?:metabolic\s+)?response)\b",
    flags=re.IGNORECASE,
)
IMAGING_EQUIVOCAL = re.compile(
    r"\b(?:equivocal|indeterminate|non[\s-]?specific|questionable|uncertain|"
    r"unclear|cannot\s+(?:be\s+)?(?:excluded?|ruled\s+out)|"
    r"(?:may|might|could)\s+(?:represent|reflect|be)|differential|versus|vs\.?|"
    r"(?:likely|probably|favou?r\w*|consistent\s+with|suggest\w*\s+of)\s+"
    r"(?:\w+\s+){0,2}?(?:benign|degenerative|inflammatory|infectious|reactive|"
    r"post[\s-]?(?:treatment|surgical|operative|radiation)|healing|fracture|"
    r"physiologic\w*|osteoarthrit\w*|arthrit\w*|hemangioma|granuloma\w*)|"
    r"flare|low[\s-]grade\s+uptake|mild\w*\s+uptake|too\s+small\s+to\s+characterize)\b",
    flags=re.IGNORECASE,
)
IMAGING_HYPOTHETICAL = re.compile(
    r"\b(?:if|whether|would|should|recommend\w*|consider\w*|plan\w*|"
    r"to\s+(?:evaluate|assess|rule\s+out|exclude|characterize)|rule\s+out|r/o|"
    r"pending|schedul\w*|follow[\s-]?up\s+(?:imaging|psma|pet|scan)\s+(?:to|in)|"
    r"family\s+history)\b",
    flags=re.IGNORECASE,
)
IMAGING_PROBABLE = re.compile(
    r"\b(?:suspicious|suspect\w*|suspicion|concern\w*|likely|probabl[ey]|"
    r"consistent\s+with|compatible\s+with|worrisome|favou?r\w*|presum\w*|"
    r"suggest\w*|possibl[ey]|highly\s+suggestive)\b",
    flags=re.IGNORECASE,
)

PROCEDURE_TERMS = {
    "radical_prostatectomy": PROSTATECTOMY_TERM,
    "radiation_therapy": RADIATION_TERM,
}


def _flatten(quote):
    return re.sub(r"\s+", " ", str(quote or "")).strip()


def _sentence_before(text, start):
    before = text[max(0, start - SENTENCE_WINDOW_CHARS):start]
    ends = list(_SENTENCE_END.finditer(before))
    return before[ends[-1].end():] if ends else before


def _agent_term(row):
    """A pattern for the model's agent name, when it is a usable word."""
    agent = _flatten(row.get("agent"))
    words = [w for w in re.findall(r"[a-z0-9-]{3,}", agent.lower())]
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def _given(text, term):
    """None when some mention of `term` reads as given; else the reason it does not."""
    hits = list(term.finditer(text))
    if not hits:
        return "no_term"
    reason = None
    for hit in hits:
        # Only the few words before the term: "No adjuvant therapy after RALP"
        # still documents the prostatectomy.
        tail = _WORDS_BEFORE.search(_sentence_before(text, hit.start())).group(0)
        cue = NOT_GIVEN_BEFORE.search(tail)
        if cue is None and NOT_GIVEN_AFTER.match(text[hit.end():hit.end() + 60]) is None:
            return None
        reason = reason or f"not_given:{cue.group(0).lower() if cue else 'post'}"
    return reason


def _screen_therapy(row, text):
    terms = [t for t in (_agent_term(row), SYSTEMIC_TERM) if t is not None]
    reasons = []
    for term in terms:
        reason = _given(text, term)
        if reason is None:
            return None
        reasons.append(reason)
    return next((r for r in reasons if r != "no_term"), "no_therapy_term")


def _screen_event_statement(text, term):
    event = NOT_AN_EVENT.search(text)
    if event is not None:
        return f"not_an_event:{event.group(0).lower()}"
    ok, reason = screen_asserted(text, term)
    return None if ok else reason


def screen_psa(row, text):
    """Reject a PSA number that is only a threshold, goal or hypothetical."""
    value = row.get("psa_ng_ml")
    if value is None:
        return None
    reason = None
    for match in re.finditer(r"(?<![\d.])\d*\.?\d+", text):
        number = float(match.group(0))
        if abs(number - float(value)) > 1e-3 * max(1.0, abs(number)):
            continue
        cue = PSA_HYPOTHETICAL.search(_sentence_before(text, match.start())[-60:])
        if cue is None:
            return None
        reason = reason or f"hypothetical_value:{cue.group(0).lower()}"
    return reason


def screen_imaging(row, text):
    """(reason, gated_certainty) for an imaging_recurrence finding."""
    certainty = row.get("certainty")
    if certainty == "equivocal":
        return "equivocal:model", None
    for pattern, label in (
        (IMAGING_HYPOTHETICAL, "hypothetical"),
        (IMAGING_NEGATED, "negated"),
        (IMAGING_EQUIVOCAL, "equivocal"),
    ):
        cue = pattern.search(text)
        if cue is not None:
            return f"{label}:{cue.group(0).lower()}", None
    if IMAGING_PROBABLE.search(text) or certainty != "definite":
        return None, "probable"
    return None, "definite"


def screen_course_finding(row):
    """Return (reason, updates). `reason` is None when the finding is usable;
    `updates` holds gate-derived fields (the imaging certainty) to apply."""
    trial = row.get("_trial_context_reason")
    if trial:
        return trial, {}
    item = row.get("item")
    text = _flatten(row.get("quote"))
    if item == "psa_value":
        return screen_psa(row, text), {}
    if item in PROCEDURE_TERMS:
        return _given(text, PROCEDURE_TERMS[item]), {}
    if item == "therapy_start":
        return _screen_therapy(row, text), {}
    if item == "therapy_current":
        return _screen_therapy(row, text), {}
    if item == "therapy_stop":
        planned = STOP_PLANNED.search(text)
        if planned is not None:
            return f"planned_stop:{planned.group(0).lower()[:40]}", {}
        if STOP_CUE.search(text) is None:
            return "no_stop_term", {}
        terms = [t for t in (_agent_term(row), SYSTEMIC_TERM) if t is not None]
        if not any(t.search(text) for t in terms):
            return "no_therapy_term", {}
        return None, {}
    if item == "bcr_statement":
        return _screen_event_statement(text, BCR_TERM), {}
    if item == "progression_event":
        reason = _screen_event_statement(text, PROGRESSION_TERM)
        if reason is None and STABLE.search(text) and not re.search(
            r"\b(?:new|progress\w*)\b", text, flags=re.IGNORECASE
        ):
            reason = "stable_disease"
        return reason, {}
    if item == "crpc_statement":
        ok, reason = screen_castration_resistant_quote(text)
        return (None if ok else reason), {}
    if item == "mcrpc_statement":
        if NON_METASTATIC.search(text):
            return "non_metastatic", {}
        if MCRPC_TERM.search(text) is None:
            return "no_mcrpc_term", {}
        ok, reason = screen_castration_resistant_quote(text)
        if not ok:
            return reason, {}
        ok, reason = screen_metastatic_quote(text)
        return (None if ok else f"metastatic:{reason}"), {}
    if item == "imaging_recurrence":
        reason, certainty = screen_imaging(row, text)
        return reason, ({"certainty": certainty} if reason is None else {})
    return f"unknown_item:{item}", {}


def usable_course_findings(findings):
    """Split a findings frame into usable rows and rejections, per patient.

    Returns ({mrn: [row, ...]}, {mrn: ["item:reason", ...]}). Rows carry any
    gate updates (imaging certainty). Every builder reads course findings
    through this, so trial-context rejections reach every endpoint.
    """
    usable, rejected = {}, {}
    for row in findings.iter_rows(named=True):
        mrn = int(row["DFCI_MRN"])
        reason, updates = screen_course_finding(row)
        if reason:
            rejected.setdefault(mrn, []).append(f"{row.get('item')}:{reason}")
            continue
        usable.setdefault(mrn, []).append({**row, **updates})
    return usable, rejected
