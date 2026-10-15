"""Deterministic quote gate for the two disease states AVPC presupposes.

Aparicio et al. (Clin Cancer Res 2013, Table 1) define aggressive-variant
prostate cancer as castration-resistant prostate carcinoma with at least one of
C1-C7, and the study enrolled metastatic CRPC. The criteria alone over-call
AVPC: bulky nodes (C4) or a neuroendocrine marker with a high LDH (C6) occur in
hormone-sensitive and non-metastatic patients too. So the two states are
extracted as findings of their own, and every grounded quote offered for one
must pass this gate before it counts.

Some criteria imply a state by definition: C2, C3 and C5 describe metastases,
and C7 is progression to castration resistance. Table 1's footnote exempts
small-cell carcinoma (C1) from castration resistance, though it must still be
metastatic. Label builders combine these with the states; this module only
screens quotes and names the implying criteria.

Both screens follow the met gate's (`met_veto.py`) conventions: flatten
whitespace, require EVERY term occurrence to survive bounded negation/hedge and
boilerplate windows, and return stable reason strings for the rejected-findings
audit.

- `screen_metastatic_quote` expands mCRPC/mHSPC/mCSPC, which the met gate's
  term set lacks, and then defers to `met_veto.screen_quote`.
  Its M1-vs-N1 check therefore applies here too, so regional pelvic nodes
  alone do not make a patient metastatic. "nmCRPC" expands to
  "non-metastatic ..." and that gate's "non" cue rejects it, as intended.
- `screen_castration_resistant_quote` accepts an explicit term (CRPC,
  castration-resistant, hormone-refractory, androgen-independent) or a
  progression stated ON castration ("PSA progression on leuprolide"). It keeps
  its own negation set because the met one is wrong for this state in both
  directions: its "non" cue would reject "non-metastatic castration-resistant",
  and its "progression to" cue would reject "progression to castration
  resistance", which here is the affirmative statement.

Bump GATE_VERSION on any change to these patterns or windows. Both run
fingerprints hash it together with the met gate's VETO_VERSION, so a changed
gate forces --overwrite instead of mixing outputs screened by two gates.
"""

import re

from tasks.longitudinal_subtype.met_veto import (
    BOILERPLATE,
    BOILERPLATE_WINDOW_CHARS,
    NEGATION_WINDOW_CHARS,
    POST_NEGATION,
    POST_WINDOW_CHARS,
    screen_quote as screen_met_quote,
)
# Re-exported: screen_metastatic_quote defers to the met gate, so the run
# fingerprints import its version from here alongside GATE_VERSION.
from tasks.longitudinal_subtype.met_veto import VETO_VERSION as MET_VETO_VERSION  # noqa: F401

GATE_VERSION = "mcrpc-gate-v1"

METASTATIC = "metastatic"
CASTRATION_RESISTANT = "castration_resistant"

# Criteria that establish a state by their own definition (Table 1).
METASTATIC_CRITERIA = frozenset({"C2", "C3", "C5"})
CASTRATION_RESISTANT_CRITERIA = frozenset({"C7"})
# Table 1, footnote a: small-cell carcinoma need not be castration-resistant.
CASTRATION_EXEMPT_CRITERIA = frozenset({"C1"})

# The met gate reads "metastatic" but not the acronyms clinicians actually
# write in an assessment ("mCRPC, on enzalutamide").
_DISEASE_ACRONYM = re.compile(r"\b(n?)m(crpc|hspc|cspc)\b", flags=re.IGNORECASE)
_ACRONYM_DISEASE = {
    "crpc": "castration-resistant prostate cancer",
    "hspc": "hormone-sensitive prostate cancer",
    "cspc": "castration-sensitive prostate cancer",
}

# Explicit castration-resistance terms. nmCRPC is castration-resistant; it is
# the metastatic screen that rejects it.
CR_TERM = re.compile(
    r"\b(?:n?m?crpc|hrpc|"
    r"castrat(?:e|ion)[\s-]+(?:resistan(?:t|ce)|refractory|recurrent)|"
    r"hormone[\s-]+(?:refractory|resistan(?:t|ce)|insensitive|independent)|"
    r"androgen[\s-]+(?:independen(?:t|ce)|refractory|resistan(?:t|ce)|"
    r"insensitive))\b",
    flags=re.IGNORECASE,
)

# Castrating therapy. Bicalutamide is left out on purpose: as monotherapy it
# does not castrate, so progression on it is not castration resistance. The
# androgen-receptor pathway inhibitors are essentially always given on ADT.
_CASTRATING_THERAPY = (
    r"(?:adt|androgen[\s-]+deprivation(?:\s+therapy)?|"
    r"(?:hormon(?:e|al)|endocrine)\s+therapy|"
    r"(?:lhrh|gnrh)\s+(?:agonist|antagonist)s?|"
    r"leuprolide|lupron|eligard|degarelix|firmagon|relugolix|orgovyx|"
    r"goserelin|zoladex|triptorelin|trelstar|orchiectomy|"
    r"abiraterone|zytiga|enzalutamide|xtandi|apalutamide|erleada|"
    r"darolutamide|nubeqa|"
    r"castrat(?:e|ion)\s+(?:levels?|testosterone|range))"
)
# One word with no clause or sentence punctuation, so a pattern cannot join
# "PSA rose." to the next sentence's "He is on ADT". Decimals ("7.2") survive.
_WORD = r"(?:[^\s.,;:!?]|\.(?=\S))+"
# Progression stated to occur ON castration. A PSA rise needs "PSA" nearby
# (checked in code) because "increased bone pain on ADT" is not progression.
PROGRESSION_ON_CASTRATION = re.compile(
    r"\b(?:(?P<rise>ris(?:e|es|ing|en)|rose|increas(?:e|ed|es|ing)|"
    r"doubl(?:ed|es|ing)|climb(?:ed|ing|s)?)|"
    r"progress(?:ed|es|ing|ion|ive)|recurr(?:ed|ence|ent)|fail(?:ed|ing|ure))\b"
    rf"(?:\s+{_WORD}){{0,4}}?\s+"
    r"(?:on|despite|while\s+(?:on|receiving|taking)|during|through|"
    r"in\s+spite\s+of)"
    rf"(?:\s+{_WORD}){{0,2}}?\s+{_CASTRATING_THERAPY}\b",
    flags=re.IGNORECASE,
)
REFRACTORY_TO_CASTRATION = re.compile(
    r"\b(?:refractory|resistan(?:t|ce)|insensitive|unresponsive)\s+to"
    rf"(?:\s+{_WORD}){{0,2}}?\s+(?:{_CASTRATING_THERAPY}|castration)\b"
    r"|\bfail(?:ed|ing|ure\s+(?:of|on))"
    rf"(?:\s+{_WORD})?\s+{_CASTRATING_THERAPY}\b",
    flags=re.IGNORECASE,
)
PSA = re.compile(r"\bpsa\b", flags=re.IGNORECASE)
PSA_WINDOW_CHARS = 30

# A therapy START or a coordinated next clause inside an inferred match means
# the progression came BEFORE castration: "PSA rose and he was started on
# ADT". The match is discarded rather than counted.
THERAPY_START = re.compile(
    r"\b(?:start\w*|initiat\w*|began|begin\w*|begun|commenc\w*|resum\w*|"
    r"restart\w*|placed|put|switch\w*|chang\w*|add(?:ed|ing)?|offer\w*|"
    r"recommend\w*|plan\w*|prescrib\w*|given|then|so|and|but|therefore|"
    r"thus|prompt\w*|led|leading|which|who)\b",
    flags=re.IGNORECASE,
)

# Negation, hedging, and non-assertion cues, scanned in a bounded window before
# each occurrence. Deliberately lacks the met gate's "non" and "progression
# to" (see the module docstring) and "developed", which is affirmative here;
# the present and future forms of "develop" still anticipate the state.
CR_NEGATION_HEDGE = re.compile(
    r"\b(?:no|not|never|without|absent|absence|denies|denied|negative|neg|"
    r"free\s+of|rule[\s-]?out|r/o|ruled\s+out|exclude[ds]?|excluding|"
    r"versus|vs|differential|ddx|consider(?:ed|ation)?|"
    r"suspicious|suspect(?:ed)?|suspicion|concern(?:ing|ed)?|"
    r"possible|possibly|probable|probably|likely|unlikely|presumed|"
    r"cannot\s+(?:be\s+)?exclude[d]?|equivocal|indeterminate|questionable|"
    r"borderline|emerging|evolving|impending|incipient|becoming|approaching|"
    r"if|whether|should|would|may|might|could|can|"
    r"evaluate|evaluation|assess|work[\s-]?up|"
    r"monitor(?:ing|ed)?|surveillance|watch(?:ing|ed)?\s+for|"
    r"develop(?:s|ing)?|before|prior\s+to|until|once|"
    r"risk\s+(?:of|for)|at\s+risk|potential|prevent\w*|delay\w*|"
    r"family\s+history|mother|father|brother|sister)\b",
    flags=re.IGNORECASE,
)

# Inferred (progression-on-therapy) matches only. A stated castration-
# sensitive state, or progression off or around therapy, means the rise is not
# castration resistance -- or at least is not safely readable as it.
CASTRATION_SENSITIVE = re.compile(
    r"\b(?:n?m?hspc|n?m?cspc|"
    r"hormone[\s-]+(?:sensitive|na[iï]ve|responsive)|"
    r"castrat(?:e|ion)[\s-]+(?:sensitive|na[iï]ve)|androgen[\s-]+sensitive)\b",
    flags=re.IGNORECASE,
)
OFF_CASTRATION = re.compile(
    r"\b(?:off|holiday|intermittent|discontinu\w*|stopp(?:ed|ing)|ceased|"
    r"held|hold(?:ing)?|withdraw\w*|flare|non[\s-]?castrat\w*|"
    r"testosterone\s+recover\w*|recover\w*\s+(?:of\s+)?testosterone)\b",
    flags=re.IGNORECASE,
)


def _flatten(quote):
    return re.sub(r"\s+", " ", str(quote or "")).strip()


def _expand_disease_acronyms(text):
    def expand(match):
        prefix = "non-metastatic" if match.group(1) else "metastatic"
        return f"{prefix} {_ACRONYM_DISEASE[match.group(2).lower()]}"

    return _DISEASE_ACRONYM.sub(expand, text)


def screen_metastatic_quote(quote):
    """Return ``(ok, reason)``: does the quote assert distant metastatic disease?

    Reasons are the met gate's: ``no_met_term`` /
    ``no_metastatic_assertion`` / ``negated_or_hedged:<cue>`` /
    ``regional_nodal_only`` / ``non_prostate_primary:<cue>`` /
    ``boilerplate:<cue>``.
    """
    return screen_met_quote(_expand_disease_acronyms(_flatten(quote)))


def _inferred_resistance(text):
    """Yield progression-on-castration matches that read as resistance."""
    for match in PROGRESSION_ON_CASTRATION.finditer(text):
        if THERAPY_START.search(match.group(0)):
            continue
        if match.group("rise") and not PSA.search(
            text[max(0, match.start() - PSA_WINDOW_CHARS):match.end()]
        ):
            continue
        yield match
    for match in REFRACTORY_TO_CASTRATION.finditer(text):
        yield match


def screen_castration_resistant_quote(quote):
    """Return ``(ok, reason)``: does the quote assert castration resistance?

    Reasons are stable strings for the rejected-findings audit:
      ``no_castration_resistance_term`` / ``negated_or_hedged:<cue>`` /
      ``boilerplate:<cue>`` / ``castration_sensitive:<cue>`` /
      ``off_castration:<cue>``
    The last two apply only to progression-on-therapy wording; an explicit
    term such as "castration-resistant" is not second-guessed by them.
    """
    text = _flatten(quote)
    hits = [(match, False) for match in CR_TERM.finditer(text)]
    hits += [(match, True) for match in _inferred_resistance(text)]
    if not hits:
        return False, "no_castration_resistance_term"
    # Every occurrence must be clean: one hedged mention anywhere in the quote
    # makes it an unsafe basis for the state.
    for hit, inferred in sorted(hits, key=lambda item: item[0].start()):
        before = text[max(0, hit.start() - NEGATION_WINDOW_CHARS):hit.start()]
        cue = CR_NEGATION_HEDGE.search(before)
        if cue is not None:
            return False, f"negated_or_hedged:{cue.group(0).lower()}"
        after = text[hit.end():hit.end() + POST_WINDOW_CHARS]
        if POST_NEGATION.match(after):
            return False, "negated_or_hedged:post_negation"
        boiler_window = text[
            max(0, hit.start() - BOILERPLATE_WINDOW_CHARS):
            hit.end() + BOILERPLATE_WINDOW_CHARS
        ]
        boiler = BOILERPLATE.search(boiler_window)
        if boiler is not None:
            return False, f"boilerplate:{boiler.group(0).lower()}"
        if inferred:
            context = text[
                max(0, hit.start() - NEGATION_WINDOW_CHARS):
                hit.end() + POST_WINDOW_CHARS
            ]
            sensitive = CASTRATION_SENSITIVE.search(context)
            if sensitive is not None:
                return False, f"castration_sensitive:{sensitive.group(0).lower()}"
            off = OFF_CASTRATION.search(context)
            if off is not None:
                return False, f"off_castration:{off.group(0).lower()}"
    return True, None


STATE_SCREENS = {
    METASTATIC: screen_metastatic_quote,
    CASTRATION_RESISTANT: screen_castration_resistant_quote,
}
