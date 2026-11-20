"""Deterministic AVPC rollup: grounded note-level facts -> patient criteria.

Paper numbering (Aparicio et al., Clin Cancer Res 2013, Table 1): C2 =
exclusively visceral metastases, C3 = predominantly lytic bone metastases.

Every finding is screened before it counts, and every number the model
reported must appear in its own quote (a node "5.4 cm" needs "5.4 cm" or
"54 mm" in the quote). The model never decides whether a criterion is met:
thresholds, pairings, exclusivity, and timing are all applied here, so
`--rebuild-labels-only` re-applies a change without new LLM calls.

States:
  metastatic   a finding passing mcrpc_gate.screen_metastatic_quote (the M1
               vs N1 veto included), or implied by C2 / C3 / C5.
  CRPC         a finding passing screen_castration_resistant_quote, or
               implied by C7.
AVPC = >= 1 criterion AND metastatic AND (CRPC OR C1); small-cell carcinoma
is exempt from castration resistance (Table 1, footnote a).

Dates use tasks/onset.py: the earliest finding is the one certainly earlier,
then the more precise. A composite criterion is dated by its LATER component,
the first date at which all of it was documented. avpc_date is the first date
at which a qualifying combination of states and criteria held.

Bump CRITERIA_VERSION on any change here; the label metadata records it.
"""

import re
from datetime import timedelta

from tasks.mcrpc_gate import (
    _CASTRATING_THERAPY,
    CASTRATION_EXEMPT_CRITERIA,
    CASTRATION_RESISTANT_CRITERIA,
    METASTATIC_CRITERIA,
    screen_castration_resistant_quote,
    screen_metastatic_quote,
)
from tasks.met_veto import BOILERPLATE
from tasks.nepc.ne_gate import (
    positive_ihc_markers,
    screen_any_asserted,
    screen_asserted,
    screen_ne_quote,
)
from tasks.onset import both_hold, date_interval, earliest
from tasks.trial_guard import trial_boilerplate_reason

CRITERIA_VERSION = "avpc-criteria-v6"

CRITERIA = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
BULKY_CM = 5.0
HIGH_GLEASON = 8
LOW_PSA_NG_ML = 10.0
HIGH_VOLUME_BONE_COUNT = 20
ULN_MULTIPLE = 2.0
# C5's PSA and bone burden must be documented at one time point: the two date
# intervals must CERTAINLY lie within this many days of each other.
C5_PAIRING_DAYS = 90
# A blastic / sclerotic description of the bone disease this close to a lytic
# one contradicts lytic predominance (C3).
C3_CONTEXT_DAYS = 180
C7_MAX_DAYS = 183
# C7 compares dates across notes, so day-level noise is tolerable but a
# year-only date is not.
C7_MIN_PRECISION = {"month", "day"}

# --- Patterns ---------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
_MEASURE = re.compile(
    r"(\d+(?:\.\d+)?)((?:\s*(?:x|×|by)\s*\d+(?:\.\d+)?)*)\s*(cm|mm)\b",
    flags=re.IGNORECASE,
)
_GLEASON_SUM = re.compile(r"\b(\d{1,2})\s*\+\s*(\d{1,2})\b")
_GLEASON_TOTAL = re.compile(
    r"\b(?:gleason|gs)\s*(?:score|grade|sum)?\s*(?:of|=|:|is)?\s*(\d{1,2})\b",
    flags=re.IGNORECASE,
)
_GRADE_GROUP = re.compile(r"\b(?:grade\s+group|gg|isup\s+grade)\s*(?:of|:)?\s*([1-5])\b", flags=re.IGNORECASE)
_GRADE_GROUP_GLEASON = {4: 8, 5: 9}

ADT_TERM = re.compile(rf"\b{_CASTRATING_THERAPY}\b", flags=re.IGNORECASE)
# First-line androgen deprivation: what "initiation of hormonal therapy" (C7)
# and "before ADT" (C5) refer to. Second-line AR-pathway inhibitors are
# hormonal but are not that start ("progressed within 3 months of
# enzalutamide" is not C7). "castration-resistant" is not a therapy.
PRIMARY_ADT = re.compile(
    r"\b(?:adt|androgen[\s-]+deprivation(?:\s+therapy)?|"
    r"(?:hormon(?:e|al)|endocrine)\s+therapy|(?:lhrh|gnrh)\s+(?:agonist|antagonist)s?|"
    r"leuprolide|lupron|eligard|degarelix|firmagon|relugolix|orgovyx|goserelin|zoladex|"
    r"triptorelin|trelstar|orchiectomy|bicalutamide|casodex|flutamide|nilutamide|"
    r"castration(?![\s-]+(?:resist|refract|sensitiv|naive)))\b",
    flags=re.IGNORECASE,
)
SECOND_LINE_HORMONAL = re.compile(
    r"\b(?:abiraterone|zytiga|enzalutamide|xtandi|apalutamide|erleada|darolutamide|"
    r"nubeqa|arpi|arsi|novel\s+(?:hormonal|antiandrogen)\w*)\b",
    flags=re.IGNORECASE,
)
ADT_PLANNED = re.compile(
    r"\b(?:plan\w*|recommend\w*|discuss\w*|offer\w*|consider\w*|will|would|"
    r"candidate|declin\w*|refus\w*|defer\w*|option\w*)\b",
    flags=re.IGNORECASE,
)

_RAPID_INTERVAL = re.compile(
    r"\b(?:within|after(?:\s+only)?|in(?:\s+only)?|by)\s+"
    r"(?:(?:about|approximately|~|only|less\s+than|under|<)\s*)?"
    r"(\d+(?:\.\d+)?|one|two|three|four|five|six)\s*(?:-|\s)?"
    r"(months?|mos?\.?|weeks?|wks?)\b",
    flags=re.IGNORECASE,
)
_WORD_NUMBER = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_PRIMARY_REFRACTORY = re.compile(
    r"\bprimar(?:y|ily)\s+(?:refractory|resistan\w+|progressive|"
    r"castrat\w+[\s-]+resistan\w+|hormone[\s-]+refractory)\b|"
    r"\bnever\s+(?:had\s+a\s+)?respon\w*\s+to\b",
    flags=re.IGNORECASE,
)
_PROGRESSION = re.compile(
    r"\b(?:progress\w*|n?m?crpc|castrat\w*[\s-]+resistan\w*|refractory|"
    r"resistan\w*|androgen[\s-]+independen\w*|rising\s+psa|"
    r"psa\s+(?:rose|rise|rising|increas\w*|progression))\b",
    flags=re.IGNORECASE,
)
_HORMONAL = re.compile(rf"\b(?:{_CASTRATING_THERAPY}|hormonal|hormone)\b", flags=re.IGNORECASE)
_PRIMARY_HORMONAL_REFRACTORY = re.compile(
    rf"(?:{_PRIMARY_REFRACTORY.pattern})(?:\s+(?:disease|prostate\s+cancer))?"
    rf"\s+(?:to|on|despite)\s+(?:(?:initial|first[- ]line)\s+)?{_HORMONAL.pattern}|"
    r"\bprimar(?:y|ily)\s+(?:castrat\w+[\s-]+resistan\w+|"
    r"hormone[\s-]+refractory)\b|"
    rf"\bnever\s+(?:had\s+a\s+)?respon\w*\s+to\s+{_HORMONAL.pattern}",
    flags=re.IGNORECASE,
)
_REFRACTORY_TO = re.compile(r"\b(?:refractory|resistan\w*|unresponsive)\s+to\b", flags=re.IGNORECASE)
_C7_HEDGE = re.compile(
    r"\b(?:if|risk|may|might|could|would|should|possible|possibly|likely|"
    r"concern\w*|suspect\w*|expected|typically|median|defined\s+as|criteria|"
    r"patients\s+(?:who|with))\b",
    flags=re.IGNORECASE,
)

SMALL_CELL = "small_cell_histology"
VISCERAL_TERM = re.compile(
    r"\b(?:(?:liver|hepatic|lung|pulmonary|adrenal|brain|cerebral|peritone\w*|"
    r"pleura\w*|visceral|pancrea\w*|splen\w*|renal|omental|mesenteric)\s+"
    r"(?:metasta\w*|mets|lesions?|masses|nodules|disease|involvement|deposits?)|"
    r"(?:metasta\w*|mets)\s+(?:to|in|involving)\s+(?:the\s+)?"
    r"(?:liver|lungs?|adrenals?|brain|peritoneum|pleura))\b",
    flags=re.IGNORECASE,
)
# A visceral finding must assert malignancy itself. "consistent with" is not
# one: "adrenal nodule consistent with adenoma" names a benign lesion.
_MET_WORD = re.compile(
    r"\b(?:metasta\w*|mets|malignan\w*|carcinoma)\b",
    flags=re.IGNORECASE,
)
EXCLUSIVE_VISCERAL = re.compile(
    r"\b(?:exclusively|only|purely|isolated)\s+(?:visceral|hepatic|liver|lung|pulmonary)\b|"
    r"\b(?:visceral|liver|hepatic)[\s-]only\b|"
    r"\b(?:visceral|liver|hepatic|lung|pulmonary)\s+(?:metasta\w*|mets)\s+only\b",
    flags=re.IGNORECASE,
)
# A single shared negative can govern both sites ("no bone or nodal mets"),
# but an absence of bone disease alone does not establish visceral exclusivity.
_C2_ABSENT_SITES = re.compile(
    r"\b(?:no|without|absence\s+of|negative\s+for)\s+"
    r"(?:evidence\s+of\s+)?(?:metastatic\s+)?"
    r"(?:bone|bony|osseous|skeletal|nodal|lymph\s*nodes?)"
    r"(?:\s+(?:metastatic\s+disease|metastases|mets|disease|involvement))?"
    r"(?:\s*(?:,|and|or|/)\s*(?:metastatic\s+)?"
    r"(?:bone|bony|osseous|skeletal|nodal|lymph\s*nodes?)"
    r"(?:\s+(?:metastatic\s+disease|metastases|mets|disease|involvement))?)*",
    flags=re.IGNORECASE,
)
BONE_TERM = re.compile(
    r"\b(?:bone|bony|osseous|skeletal|osteo\w*|vertebra\w*|spine|spinal|"
    r"sacr\w*|iliac|ribs?|femur|femoral|humer\w*|calvari\w*|skull)\b",
    flags=re.IGNORECASE,
)
LYTIC = re.compile(r"\b(?:osteo)?lytic\b", flags=re.IGNORECASE)
BLASTIC = re.compile(r"\b(?:osteo)?(?:blastic|sclerotic)\b", flags=re.IGNORECASE)
LYTIC_PREDOMINANT = re.compile(
    r"\b(?:predominantly|predominately|mostly|primarily|largely|chiefly|purely|"
    r"exclusively|entirely)\s+(?:osteo)?lytic\b|\b(?:osteo)?lytic[\s-]+predominan\w*",
    flags=re.IGNORECASE,
)
# Lytic disease described as the burden, not one lesion: plural lesions, or a
# plural subject described as lytic ("multiple lytic", "lesions are lytic").
LYTIC_BURDEN = re.compile(
    r"\b(?:osteo)?lytic\s+(?:[a-z-]+\s+){0,3}?"
    r"(?:lesions|metastases|mets|deposits|foci|disease|involvement)\b|"
    r"\b(?:lesions|metastases|mets|deposits|foci)\s+(?:are|were|appear\w*|remain\w*)\s+"
    r"(?:[a-z-]+\s+){0,2}?(?:osteo)?lytic\b|"
    r"\b(?:multiple|numerous|several|innumerable|multifocal|diffuse|widespread|extensive)\s+"
    r"(?:[a-z-]+\s+){0,3}?(?:osteo)?lytic\b",
    flags=re.IGNORECASE,
)
BLASTIC_PREDOMINANT = re.compile(
    r"\b(?:predominantly|predominately|mostly|primarily|largely|chiefly|purely|"
    r"exclusively|entirely|diffuse(?:ly)?)\s+(?:osteo)?(?:blastic|sclerotic)\b|"
    r"\b(?:osteo)?(?:blastic|sclerotic)[\s-]+predominan\w*",
    flags=re.IGNORECASE,
)
HIGH_VOLUME = re.compile(
    r"\b(?:innumerable|too\s+numerous\s+to\s+count|tntc|super[\s-]?scan|countless)\b",
    flags=re.IGNORECASE,
)
# "innumerable pulmonary nodules" in a bone quote is not bone burden.
_HIGH_VOLUME_NOT_BONE = re.compile(
    r"\s*(?:[a-z-]+\s+){0,2}?(?:pulmonary|lungs?|liver|hepatic|nodal|lymph|nodules|"
    r"adrenal|brain|peritone\w*|pleura\w*)\b",
    flags=re.IGNORECASE,
)
# A number counts bone lesions only when it is attached to a lesion noun, so
# "PSA greater than 20" or "25 mm lesion" is never a count of 20+ lesions.
_LESION_COUNT = re.compile(
    r"(?:(?:>|≥|>=)\s*|\b(?:more\s+than|greater\s+than|over|at\s+least|approximately|"
    r"about|~)\s*)?(?<![\w.])(\d+)\+?\s+(?:(?!cm\b|mm\b)[a-z-]+\s+){0,3}?"
    r"(?:lesions|metastases|mets|foci|sites|deposits|areas)\b",
    flags=re.IGNORECASE,
)
# A note-dated finding phrased about the past cannot be placed in time: its
# note date is when it was retold, not when it held.
RETROSPECTIVE = re.compile(
    r"\b(?:at\s+(?:the\s+time\s+of\s+)?(?:initial\s+)?(?:diagnosis|presentation|dx)|"
    r"initial(?:ly)?|original(?:ly)?|previous(?:ly)?|history\s+of|back\s+in|"
    r"(?:in|from|since)\s+(?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)

NODE_TERM = re.compile(
    r"\b(?:lymph\s*nodes?|nodes?|nodal|(?:lymph)?adenopathy|lad|conglomerate)\b",
    flags=re.IGNORECASE,
)
MASS_TERM = re.compile(r"\b(?:mass(?:es)?|tumou?r|lesion)\b", flags=re.IGNORECASE)
PELVIC_SITE = re.compile(
    r"\b(?:prostat\w*|pelvi\w*|bladder|rect\w*|seminal|perirectal|presacral)\b",
    flags=re.IGNORECASE,
)
# Sites that, when nearer a measurement than a pelvic / nodal word, own it:
# "CT abdomen/pelvis: 6.1 cm liver mass" is a liver mass, not C4.
_NON_PELVIC_SITE = re.compile(
    r"\b(?:liver|hepatic|lungs?|pulmonary|adrenals?|brain|cerebral|kidneys?|renal|"
    r"spleen|splenic|pancrea\w*|pleura\w*|peritone\w*|omentum|omental|mesenter\w*|"
    r"bone|bony|osseous|vertebra\w*|spine|spinal|ribs?|femur|femoral|humer\w*|skull|"
    r"calvari\w*|chest\s+wall|thyroid)\b",
    flags=re.IGNORECASE,
)
_NON_NODAL_SITE = re.compile(
    rf"{_NON_PELVIC_SITE.pattern}|\b(?:prostat\w*|bladder|rect(?:um|al))\b",
    flags=re.IGNORECASE,
)
# Exam names mention the pelvis without locating anything there.
_EXAM_REGION = re.compile(
    r"\b(?:(?:chest\s*,?\s*)?(?:abdomen|abd)\s*,?\s*(?:and|/|&|-)?\s*pelvis|"
    r"(?:ct|mri?|pet)\s*(?:of\s+the\s+)?pelvis)\b",
    flags=re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"\.(?!\d)|(?<!\d)\.|[;!?\n]")

PSA_TERM = re.compile(r"\bpsa\b", flags=re.IGNORECASE)
PSA_CONTEXT_CUES = {
    "initial_presentation_pre_adt": re.compile(
        r"\b(?:diagnos\w*|dx|present\w*|initial\w*|baseline|pre-?treatment|"
        r"before|prior|at\s+the\s+time|biopsy)\b",
        flags=re.IGNORECASE,
    ),
    "symptomatic_crpc_progression": re.compile(
        r"\b(?:progress\w*|rising|rose|symptom\w*|pain|n?m?crpc|castrat\w*|"
        r"refractory|resistan\w*)\b",
        flags=re.IGNORECASE,
    ),
}

C6_TISSUE_MARKERS = {"chromogranin", "chromogranin a", "synaptophysin"}
SERUM_MARKER = re.compile(
    r"\b(?:chromogranin(?:\s+a)?|cga|pro-?grp|grp|gastrin[\s-]+releasing\s+peptide)\b",
    flags=re.IGNORECASE,
)
ELEVATED = re.compile(
    r"\b(?:elevat\w*|high|increas\w*|abnormal|rising|rose|above)\b|>",
    flags=re.IGNORECASE,
)
LAB_TERMS = {
    "elevated_ldh": re.compile(r"\b(?:ldh|lactate\s+dehydrogenase)\b", flags=re.IGNORECASE),
    "elevated_cea": re.compile(r"\b(?:cea|carcinoembryonic\s+antigen)\b", flags=re.IGNORECASE),
}
_ULN_MULTIPLE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:x|×|times|-?fold)\s*(?:the\s+)?"
    r"(?:uln|upper\s+limit|normal|institutional)|"
    r"\b(twice|double|triple)\s+(?:the\s+)?(?:uln|upper\s+limit|normal)",
    flags=re.IGNORECASE,
)
_WORD_MULTIPLE = {"twice": 2.0, "double": 2.0, "triple": 3.0}
HYPERCALCEMIA = re.compile(r"\bhypercalc\w*", flags=re.IGNORECASE)
MALIGNANCY_CUE = re.compile(
    r"\b(?:malignan\w*|humoral|pthrp|pth-?rp|metasta\w*|tumou?r|cancer|"
    r"carcinoma|osteolytic|lytic)\b",
    flags=re.IGNORECASE,
)

# --- Number support -----------------------------------------------------------------


def _numbers(text):
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
    return [float(n) for n in _NUMBER.findall(text)]


def _close(a, b, tol=1e-3):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def number_in_quote(value, quote):
    if value is None:
        return False
    return any(_close(float(value), n) for n in _numbers(quote))


def measurements_cm(quote):
    """Every dimension of every measurement in the quote, in cm."""
    dims = []
    for match in _MEASURE.finditer(quote or ""):
        values = [match.group(1)] + re.findall(r"\d+(?:\.\d+)?", match.group(2) or "")
        scale = 0.1 if match.group(3).lower() == "mm" else 1.0
        dims.extend(float(v) * scale for v in values)
    return dims


def size_in_quote(size_cm, quote):
    if size_cm is None:
        return False
    return any(_close(float(size_cm), d, tol=1e-2) for d in measurements_cm(quote))


def metastatic_assertion_reason(quote, met_word):
    """None if some malignancy word in `quote` is asserted, else a reason.

    Presence alone is not enough: "likely benign; no metastatic disease" and
    "not consistent with metastases" both contain one.
    """
    # Assertion only: the visceral-site screen runs the site/trial checks.
    ok, reason = screen_any_asserted(quote or "", met_word, context=False)
    if ok:
        return None
    return "no_metastatic_assertion" if reason == "no_term" else f"no_metastatic_assertion:{reason}"


def gleason_in_quote(total, quote):
    if total is None:
        return False
    total = int(total)
    candidates = {int(a) + int(b) for a, b in _GLEASON_SUM.findall(quote or "")}
    candidates.update(int(n) for n in _GLEASON_TOTAL.findall(quote or ""))
    for group in _GRADE_GROUP.findall(quote or ""):
        if int(group) in _GRADE_GROUP_GLEASON and total >= HIGH_GLEASON:
            candidates.add(total)
    return total in candidates


def _uln_ratio(row):
    """Value / ULN when both are in the quote, else a stated multiple of ULN."""
    quote = row.get("quote") or ""
    value, uln = row.get("value"), row.get("uln")
    if (
        value is not None and uln is not None and float(uln) > 0
        and number_in_quote(value, quote) and number_in_quote(uln, quote)
    ):
        return float(value) / float(uln)
    best = None
    for number, word in _ULN_MULTIPLE.findall(quote):
        ratio = float(number) if number else _WORD_MULTIPLE[word.lower()]
        best = ratio if best is None else max(best, ratio)
    return best


# --- Per-finding screens ------------------------------------------------------------


def _reject(rejections, row, reason):
    rejections.append(f"{row.get('item')}:{reason}")


def _screen_rapid_progression(quote):
    if BOILERPLATE.search(quote):
        return "boilerplate"
    hedge = _C7_HEDGE.search(quote)
    if hedge is not None:
        return f"negated_or_hedged:{hedge.group(0).lower()}"
    if SECOND_LINE_HORMONAL.search(quote) and not PRIMARY_ADT.search(quote):
        return "c7_second_line_hormonal_therapy"
    if _PRIMARY_REFRACTORY.search(quote):
        if not _PRIMARY_HORMONAL_REFRACTORY.search(quote):
            return "no_primary_refractoriness_to_hormonal_therapy"
        ok, reason = screen_asserted(quote, _PRIMARY_REFRACTORY)
        return None if ok else reason
    if not (_PROGRESSION.search(quote) and _HORMONAL.search(quote)):
        return "no_progression_on_hormonal_therapy"
    if _REFRACTORY_TO.search(quote):
        ok, reason = screen_castration_resistant_quote(quote)
        if not ok:
            return reason
    ok, reason = screen_asserted(quote, _PROGRESSION)
    if not ok:
        return reason
    # Screen the interval too: in "progression within 4 months of ADT was not
    # observed", the post-negation is too far from the word "progression".
    if _RAPID_INTERVAL.search(quote):
        ok, reason = screen_asserted(quote, _RAPID_INTERVAL)
        if not ok:
            return reason
    for number, unit in _RAPID_INTERVAL.findall(quote):
        amount = _WORD_NUMBER.get(number.lower()) or float(number)
        months = amount / 4.345 if unit.lower().startswith("w") else amount
        if months <= 6:
            return None
    return "no_interval_within_6_months"


def _screen_serum_marker(quote):
    if not SERUM_MARKER.search(quote):
        return "no_serum_ne_marker"
    if not ELEVATED.search(quote):
        return "serum_marker_not_elevated"
    ok, reason = screen_asserted(quote, SERUM_MARKER)
    return None if ok else reason


def _blank(match):
    return " " * len(match.group(0))


def _nearest(text, span, wanted, other):
    """True when the term nearest `span` in its sentence is `wanted`."""
    start = max((m.end() for m in _SENTENCE_BREAK.finditer(text, 0, span[0])), default=0)
    stop = next((m.start() for m in _SENTENCE_BREAK.finditer(text, span[1])), len(text))
    sentence = text[start:stop]

    def distance(pattern):
        best = None
        for m in pattern.finditer(sentence):
            a, b = m.start() + start, m.end() + start
            d = span[0] - b if b <= span[0] else (a - span[1] if a >= span[1] else 0)
            best = d if best is None else min(best, d)
        return best

    near_wanted, near_other = distance(wanted), distance(other)
    return near_wanted is not None and (near_other is None or near_wanted < near_other)


def _screen_measured_site(item, size_cm, quote):
    """C4 needs the reported measurement to belong to a node (or a prostate /
    pelvic mass), not to a liver or bone lesion mentioned in the same quote."""
    text = _EXAM_REGION.sub(_blank, quote)
    if item == "bulky_lymphadenopathy":
        wanted, other, reason = NODE_TERM, _NON_NODAL_SITE, "c4_measurement_not_nodal"
    else:
        wanted, other, reason = PELVIC_SITE, _NON_PELVIC_SITE, "c4_measurement_not_prostate_pelvic"
    for match in _MEASURE.finditer(text):
        values = [match.group(1)] + re.findall(r"\d+(?:\.\d+)?", match.group(2) or "")
        scale = 0.1 if match.group(3).lower() == "mm" else 1.0
        if not any(_close(float(size_cm), float(v) * scale, tol=1e-2) for v in values):
            continue
        if _nearest(text, match.span(), wanted, other):
            return None
    return reason


def screen_finding(row):
    """Return None when the finding can count, else a rejection reason."""
    item = row.get("item")
    quote = row.get("quote") or ""
    trial_reason = row.get("_trial_context_reason") or trial_boilerplate_reason(quote)
    if trial_reason:
        return trial_reason
    if item in {"metastatic_disease", "bone_metastasis"}:
        ok, reason = screen_metastatic_quote(quote)
        return None if ok else reason
    if item == "castration_resistance":
        ok, reason = screen_castration_resistant_quote(quote)
        return None if ok else reason
    if item == "adt_start":
        if not ADT_TERM.search(quote):
            return "no_adt_term"
        if ADT_PLANNED.search(quote):
            return "adt_not_started:" + ADT_PLANNED.search(quote).group(0).lower()
        ok, reason = screen_asserted(quote, ADT_TERM)
        return None if ok else reason
    if item == "rapid_progression_on_adt":
        return _screen_rapid_progression(quote)
    if item == SMALL_CELL:
        ok, reason = screen_ne_quote(quote, SMALL_CELL)
        return None if ok else reason
    if item == "visceral_metastasis":
        reason = metastatic_assertion_reason(quote, _MET_WORD)
        if reason:
            return reason
        ok, reason = screen_asserted(quote, VISCERAL_TERM)
        return None if ok else ("no_visceral_term" if reason == "no_term" else reason)
    if item in {"bulky_lymphadenopathy", "bulky_prostate_pelvic_mass"}:
        term = NODE_TERM if item == "bulky_lymphadenopathy" else MASS_TERM
        if item == "bulky_prostate_pelvic_mass" and not PELVIC_SITE.search(quote):
            return "no_prostate_pelvic_site"
        if not size_in_quote(row.get("size_cm"), quote):
            return "size_not_in_quote"
        ok, reason = screen_asserted(quote, term)
        if not ok:
            return reason
        return _screen_measured_site(item, row.get("size_cm"), quote)
    if item == "gleason_high_grade":
        return None if gleason_in_quote(row.get("gleason_total"), quote) else "gleason_not_in_quote"
    if item == "low_psa":
        if not PSA_TERM.search(quote):
            return "no_psa_term"
        if not number_in_quote(row.get("psa_ng_ml"), quote):
            return "psa_not_in_quote"
        return None
    if item == "ne_marker":
        if row.get("specimen") == "serum":
            return _screen_serum_marker(quote)
        markers, reason = positive_ihc_markers(quote)
        if set(markers) & C6_TISSUE_MARKERS:
            return None
        if row.get("specimen") is None and _screen_serum_marker(quote) is None:
            return None
        return reason or "no_chromogranin_or_synaptophysin"
    if item in LAB_TERMS:
        if not LAB_TERMS[item].search(quote):
            return "no_lab_term"
        return None
    if item == "malignant_hypercalcemia":
        if not MALIGNANCY_CUE.search(quote):
            return "no_malignancy_attribution"
        ok, reason = screen_asserted(quote, HYPERCALCEMIA)
        return None if ok else reason
    return f"unknown_item:{item}"


# --- Criteria -----------------------------------------------------------------------


def _interval(row):
    return date_interval(row.get("event_date"), row.get("date_precision"))


def _later(*rows):
    """Combine prerequisites without inventing a date or its precision."""
    return {
        **rows[0],
        **(both_hold(rows) or {"event_date": None, "date_precision": "unknown"}),
        "date_source": ("unknown" if any(not r.get("event_date") for r in rows)
                        else "stated" if all(r.get("date_source") == "stated" for r in rows)
                        else "note_date"),
    }


def _gap_days(a, b):
    """Smallest distance in days between two findings' date intervals, or None."""
    ia, ib = _interval(a), _interval(b)
    if ia is None or ib is None:
        return None
    if ia[1] < ib[0]:
        return (ib[0] - ia[1]).days
    if ib[1] < ia[0]:
        return (ia[0] - ib[1]).days
    return 0


def lesion_counts(quote):
    """Numbers in the quote written as a count of lesions ("24 osseous lesions",
    "more than 20 metastases")."""
    return [int(n) for n in _LESION_COUNT.findall(quote or "")]


_COUNT_CLAUSE_BREAK = re.compile(r"[.;!?,]|\b(?:and|but|while|whereas)\b", re.IGNORECASE)
_NON_BONE_COUNT_SITE = re.compile(
    r"\b(?:liver|hepatic|lungs?|pulmonary|adrenal\w*|brain|cerebral|renal|kidneys?|"
    r"pleura\w*|peritone\w*|pancrea\w*|splen\w*|spleen|visceral|soft[\s-]+tissue|"
    r"skin|cutaneous|subcutaneous|muscle|muscular|nodes?|nodal|lymph\w*)\b", re.IGNORECASE,
)


def bone_lesion_counts(quote):
    """Count lesions only in a bone clause, never borrow another site's count."""
    quote = quote or ""
    for match in _LESION_COUNT.finditer(quote):
        before = _COUNT_CLAUSE_BREAK.split(quote[:match.start()])[-1]
        after = _COUNT_CLAUSE_BREAK.split(quote[match.end():])[0]
        phrase = match.group(0) + after
        if _NON_BONE_COUNT_SITE.search(phrase):
            continue
        context = before + phrase
        if (BONE_TERM.search(phrase) or
                not _NON_BONE_COUNT_SITE.search(context)
                and (BONE_TERM.search(context) or not _NON_BONE_COUNT_SITE.search(quote))):
            yield int(match.group(1))


def _is_high_volume(row):
    """>= 20 bone lesions. The model's `count` must be a number the quote writes
    as a lesion count; otherwise only "innumerable" / "too numerous to count" /
    "superscan" / "countless" qualifies. The model never counts lesions itself,
    and "multiple", "diffuse" or "extensive" alone are not >= 20."""
    quote = row.get("quote") or ""
    if any(n >= HIGH_VOLUME_BONE_COUNT for n in bone_lesion_counts(quote)):
        return True
    for match in HIGH_VOLUME.finditer(quote):
        if _HIGH_VOLUME_NOT_BONE.match(quote, match.end()):
            continue
        ok, _ = screen_asserted(quote, HIGH_VOLUME)
        return ok
    return False


def _criterion(row, quote=None, components=None):
    return {"row": row, "quote": quote or row["quote"], "components": components or {}}


def _earliest_criterion(candidates):
    first = earliest([candidate["row"] for candidate in candidates])
    return next((candidate for candidate in candidates if candidate["row"] is first), None)


def _visceral_exclusivity_in_quote(quote):
    if EXCLUSIVE_VISCERAL.search(quote):
        return True
    absent_sites = " ".join(match.group(0) for match in _C2_ABSENT_SITES.finditer(quote))
    return bool(BONE_TERM.search(absent_sites) and NODE_TERM.search(absent_sites))


def evaluate_c2(visceral, bone_evidence, rejections, nodal_evidence=()):
    """Exclusively visceral metastases without prior/concurrent bone or nodal disease."""
    candidates = []
    for row in visceral:
        if row.get("exclusively_visceral") is not True:
            continue
        if not _visceral_exclusivity_in_quote(row["quote"]):
            _reject(rejections, row, "c2_exclusivity_not_in_quote")
            continue
        interval = _interval(row)
        earlier_bone = [
            b for b in bone_evidence
            if interval is None or _interval(b) is None or _interval(b)[0] <= interval[1]
        ]
        if earlier_bone:
            _reject(rejections, row, "c2_bone_disease_on_or_before")
            continue
        earlier_nodes = [
            n for n in nodal_evidence
            if interval is None or _interval(n) is None or _interval(n)[0] <= interval[1]
        ]
        if earlier_nodes:
            _reject(rejections, row, "c2_nodal_disease_on_or_before")
            continue
        candidates.append(row)
    first = earliest(candidates)
    return _criterion(first) if first else None


def _blastic_descriptions(bone_evidence):
    """Bone findings describing the disease as blastic / sclerotic or mixed,
    without stating lytic predominance."""
    out = []
    for row in bone_evidence:
        quote = row.get("quote") or ""
        if LYTIC_PREDOMINANT.search(quote) or not BLASTIC.search(quote):
            continue
        ok, _ = screen_asserted(quote, BLASTIC)
        if ok:
            out.append(row)
    return out


def evaluate_c3(bone, rejections, bone_evidence=None):
    """Predominantly lytic bone metastases.

    The quote must describe lytic BURDEN (explicit predominance, or plural lytic
    lesions); one lytic lesion is not predominance. The patient's other bone
    descriptions within C3_CONTEXT_DAYS are then checked: any blastic, sclerotic
    or mixed description contradicts plural-lytic wording, and an explicit
    "predominantly / diffusely sclerotic" contradicts even "predominantly lytic".
    """
    blastic = _blastic_descriptions(bone if bone_evidence is None else bone_evidence)
    candidates = []
    for row in bone:
        if row.get("lytic_predominant") is not True:
            continue
        quote = row["quote"]
        if not LYTIC.search(quote):
            _reject(rejections, row, "c3_lytic_not_in_quote")
            continue
        explicit = LYTIC_PREDOMINANT.search(quote) is not None
        if BLASTIC.search(quote) and not explicit:
            _reject(rejections, row, "c3_lytic_predominance_not_stated")
            continue
        if not explicit and not LYTIC_BURDEN.search(quote):
            _reject(rejections, row, "c3_single_lytic_lesion")
            continue
        contradictions = [
            b for b in blastic
            if b is not row
            and (explicit is False or BLASTIC_PREDOMINANT.search(b["quote"]))
            and (_gap_days(row, b) is None or _gap_days(row, b) <= C3_CONTEXT_DAYS)
        ]
        if contradictions:
            _reject(rejections, row, "c3_contradicted_by_blastic_description")
            continue
        candidates.append(row)
    first = earliest(candidates)
    return _criterion(first) if first else None


def evaluate_c4(nodes, masses, gleason):
    bulky_nodes = [r for r in nodes if (r.get("size_cm") or 0) >= BULKY_CM]
    candidates = []
    first = earliest(bulky_nodes)
    if first is not None:
        candidates.append(_criterion(first, components={"c4_size_quote": first["quote"]}))
    bulky_masses = [r for r in masses if (r.get("size_cm") or 0) >= BULKY_CM]
    high_grade = [r for r in gleason if (r.get("gleason_total") or 0) >= HIGH_GLEASON]
    mass, grade = earliest(bulky_masses), earliest(high_grade)
    if mass is not None and grade is not None:
        candidates.append(_criterion(
            _later(mass, grade),
            quote=f"{mass['quote']} | {grade['quote']}",
            components={"c4_size_quote": mass["quote"], "c4_gleason_quote": grade["quote"]},
        ))
    return _earliest_criterion(candidates)


def _certain_gap_days(a, b):
    """Largest possible distance in days between two findings' date intervals,
    or None: a year-precision date is never certainly within 90 days."""
    ia, ib = _interval(a), _interval(b)
    if ia is None or ib is None:
        return None
    return max((ib[1] - ia[0]).days, (ia[1] - ib[0]).days)


def first_adt_start(adt):
    """The earliest first-line ADT start (any date source), or None. A note-dated
    start is only an upper bound on the true start."""
    return earliest([r for r in adt if PRIMARY_ADT.search(r.get("quote") or "")])


def _stated(row):
    return row is not None and row.get("date_source") == "stated"


def _c5_psa_context(row, anchor, rejections):
    """The C5 time point a low PSA was measured at, or None.

    "initial_presentation_pre_adt" must not fall after the first ADT start; a
    note-dated one is only placeable when its note precedes a stated first ADT
    start (a later note retelling the diagnostic PSA carries the wrong date).
    "symptomatic_crpc_progression" must not fall before a stated first ADT start.
    With no model context, a stated PSA certainly before a stated first ADT start
    is pre-ADT.
    """
    context = row.get("psa_context")
    iv = _interval(row)
    anchor_iv = _interval(anchor) if anchor is not None else None
    if iv is None:
        return None
    if context in PSA_CONTEXT_CUES and not PSA_CONTEXT_CUES[context].search(row["quote"]):
        _reject(rejections, row, "c5_psa_context_not_in_quote")
        return None
    if context == "initial_presentation_pre_adt":
        if anchor_iv and iv[0] > anchor_iv[1]:
            _reject(rejections, row, "c5_pre_adt_psa_after_adt_start")
            return None
        if not _stated(row) and not (_stated(anchor) and iv[1] < anchor_iv[0]):
            _reject(rejections, row, "c5_presentation_psa_undated")
            return None
        return context
    if context == "symptomatic_crpc_progression":
        if _stated(anchor) and iv[1] < anchor_iv[0]:
            _reject(rejections, row, "c5_castrate_psa_before_adt_start")
            return None
        if not _stated(row) and RETROSPECTIVE.search(row["quote"]):
            _reject(rejections, row, "c5_psa_date_retrospective")
            return None
        return context
    if _stated(row) and _stated(anchor) and iv[1] < anchor_iv[0]:
        return "inferred_pre_adt"
    return None


def evaluate_c5(psa, bone, rejections, is_high_volume=None, adt=()):
    """Low PSA (<= 10) and high-volume (>= 20) bone metastases at ONE time point:
    initial presentation before ADT, or progression in the castrate setting.
    The two findings' dates must certainly lie within C5_PAIRING_DAYS of each
    other, so a year-only date never pairs. `is_high_volume` lets
    tasks/avpc_adjacent relax the volume test; the default is the paper's."""
    is_high_volume = is_high_volume or _is_high_volume
    anchor = first_adt_start(adt)
    low = []
    for row in psa:
        value = row.get("psa_ng_ml")
        if value is None or value > LOW_PSA_NG_ML:
            continue
        context = _c5_psa_context(row, anchor, rejections)
        if context is not None:
            low.append((row, context))
    heavy = []
    for row in bone:
        if not is_high_volume(row):
            continue
        if not _stated(row) and RETROSPECTIVE.search(row["quote"]):
            _reject(rejections, row, "c5_bone_date_retrospective")
            continue
        heavy.append(row)
    pairs = []
    for p, context in low:
        for b in heavy:
            gap = _certain_gap_days(p, b)
            if gap is not None and gap <= C5_PAIRING_DAYS:
                pairs.append((p, b, context, gap))
    if not pairs:
        if heavy:
            for p, _ in low:
                _reject(rejections, p, "c5_no_concurrent_high_volume_bone")
        return None

    def completed(pair):
        last = _later(pair[0], pair[1])
        return last["event_date"], last.get("source_note_date") or ""

    p, b, context, gap = min(pairs, key=completed)
    return _criterion(
        _later(b, p),
        quote=f"{b['quote']} | {p['quote']}",
        components={
            "c5_psa_quote": p["quote"],
            "c5_bone_quote": b["quote"],
            "c5_psa_date": p.get("event_date"),
            "c5_bone_date": b.get("event_date"),
            "c5_psa_context": context,
            "c5_gap_days": gap,
        },
    )


def evaluate_c6(markers, labs, hypercalcemia):
    lab_hits = [r for r in labs if (_uln_ratio(r) or 0) >= ULN_MULTIPLE] + list(hypercalcemia)
    marker, lab = earliest(markers), earliest(lab_hits)
    if marker is None or lab is None:
        return None
    return _criterion(
        _later(marker, lab),
        quote=f"{marker['quote']} | {lab['quote']}",
        components={"c6_marker_quote": marker["quote"], "c6_lab_quote": lab["quote"]},
    )


def evaluate_c7(rapid, adt, crpc, rejections=None):
    """<= 6 months from the FIRST hormonal therapy to castration resistance.

    The anchor is the earliest first-line ADT start of any date source, never a
    later precise start or a second-line agent's start: if the earliest is not
    stated to month precision, or castration resistance is documented before
    it, the interval is not computed. The onset is the earliest precisely
    dated CRPC finding certainly after the anchor (an upper bound on onset).
    """
    rejections = [] if rejections is None else rejections
    explicit = earliest(rapid)
    candidates = [_criterion(explicit)] if explicit is not None else []
    start = first_adt_start(adt)
    if start is None:
        return _earliest_criterion(candidates)
    if not _stated(start) or start.get("date_precision") not in C7_MIN_PRECISION:
        _reject(rejections, start, "c7_first_adt_start_not_dated")
        return _earliest_criterion(candidates)
    start_iv = _interval(start)
    if any(_interval(r) is not None and _interval(r)[1] < start_iv[0] for r in crpc):
        _reject(rejections, start, "c7_crpc_before_first_adt_start")
        return _earliest_criterion(candidates)
    onset = earliest([
        r for r in crpc
        if r.get("date_precision") in C7_MIN_PRECISION and _interval(r)[0] > start_iv[1]
    ])
    if onset is None:
        return _earliest_criterion(candidates)
    onset_iv = _interval(onset)
    if onset_iv[1] - start_iv[0] > timedelta(days=C7_MAX_DAYS):
        return _earliest_criterion(candidates)
    interval = (onset_iv[0] - start_iv[0]).days
    candidates.append(_criterion(
        onset,
        quote=f"{start['quote']} | {onset['quote']}",
        components={
            "c7_adt_start_quote": start["quote"],
            "c7_crpc_quote": onset["quote"],
            "c7_interval_days": interval,
        },
    ))
    return _earliest_criterion(candidates)


# --- Patient --------------------------------------------------------------------------

COMPONENT_COLUMNS = (
    "c4_size_quote", "c4_gleason_quote", "c5_psa_quote", "c5_bone_quote",
    "c5_psa_date", "c5_bone_date", "c5_psa_context", "c5_gap_days", "c6_marker_quote", "c6_lab_quote", "c7_adt_start_quote", "c7_crpc_quote",
    "c7_interval_days",
)


def _state(direct, implied, criteria):
    """(row-like dict, source) for a state: the earliest of a direct quote and
    any criterion that implies the state by definition."""
    candidates = []
    if direct is not None:
        candidates.append(({**direct, "_source": "quote"}))
    for name in sorted(implied):
        if name in criteria:
            candidates.append({
                **criteria[name]["row"],
                "quote": criteria[name]["quote"],
                "_source": f"implied:{name}",
            })
    first = earliest(candidates)
    if first is None:
        return None, None
    return first, first["_source"]


def evaluate_patient(rows):
    """Return (label fields, rejection reasons) for one patient's grounded findings."""
    rejections = []
    by_item = {}
    nodal_evidence = []
    for row in rows:
        reason = screen_finding(row)
        # Regional nodal metastases do not establish M1, but still contradict
        # exclusively visceral disease. Do not treat a negated node mention as
        # a contradiction merely because another site in the quote is positive.
        if reason in {None, "regional_nodal_only"} and row["item"] in {
            "metastatic_disease", "bulky_lymphadenopathy", "visceral_metastasis",
        }:
            nodal_asserted, _ = screen_asserted(row["quote"], NODE_TERM)
            metastatic, met_reason = screen_metastatic_quote(row["quote"])
            if nodal_asserted and (metastatic or met_reason == "regional_nodal_only"):
                nodal_evidence.append(row)
        if reason:
            _reject(rejections, row, reason)
            continue
        by_item.setdefault(row["item"], []).append(row)
    get = lambda item: by_item.get(item, [])  # noqa: E731

    metastatic_direct = earliest(get("metastatic_disease") + get("bone_metastasis") + get("visceral_metastasis"))
    crpc_direct = earliest(get("castration_resistance"))
    bone_evidence = get("bone_metastasis") + [
        r for r in get("metastatic_disease")
        if BONE_TERM.search(r["quote"]) or BONE_TERM.search(r.get("sites") or "")
    ]

    criteria = {}
    small_cell = earliest(get(SMALL_CELL))
    if small_cell is not None:
        criteria["C1"] = _criterion(small_cell)
    for name, result in (
        ("C2", evaluate_c2(get("visceral_metastasis"), bone_evidence, rejections, nodal_evidence)),
        ("C3", evaluate_c3(get("bone_metastasis"), rejections, bone_evidence)),
        ("C4", evaluate_c4(get("bulky_lymphadenopathy"), get("bulky_prostate_pelvic_mass"),
                           get("gleason_high_grade"))),
        ("C5", evaluate_c5(get("low_psa"), get("bone_metastasis"), rejections,
                           adt=get("adt_start"))),
        ("C6", evaluate_c6(get("ne_marker"), get("elevated_ldh") + get("elevated_cea"),
                           get("malignant_hypercalcemia"))),
        ("C7", evaluate_c7(get("rapid_progression_on_adt"), get("adt_start"),
                           get("castration_resistance"), rejections)),
    ):
        if result is not None:
            criteria[name] = result

    metastatic, metastatic_source = _state(metastatic_direct, METASTATIC_CRITERIA, criteria)
    crpc, crpc_source = _state(crpc_direct, CASTRATION_RESISTANT_CRITERIA, criteria)

    fields = {
        "is_metastatic": metastatic is not None,
        "metastatic_quote": metastatic and metastatic["quote"],
        "metastatic_date": metastatic and metastatic.get("event_date"),
        "metastatic_source": metastatic_source,
        "metastatic_date_precision": metastatic and metastatic.get("date_precision"),
        "metastatic_date_source": metastatic and metastatic.get("date_source"),
        "is_castration_resistant": crpc is not None,
        "crpc_quote": crpc and crpc["quote"],
        "crpc_date": crpc and crpc.get("event_date"),
        "crpc_source": crpc_source,
        "crpc_date_precision": crpc and crpc.get("date_precision"),
        "crpc_date_source": crpc and crpc.get("date_source"),
        "is_mcrpc": metastatic is not None and crpc is not None,
    }
    for name in CRITERIA:
        key = name.lower()
        result = criteria.get(name)
        row = result and result["row"]
        fields[f"{key}_met"] = result is not None
        fields[f"{key}_quote"] = result and result["quote"]
        fields[f"{key}_date"] = row and row.get("event_date")
        fields[f"{key}_date_precision"] = row and row.get("date_precision")
        fields[f"{key}_source_note_date"] = row and row.get("source_note_date")
    for column in COMPONENT_COLUMNS:
        fields[column] = None
    for result in criteria.values():
        fields.update(result["components"])

    met_names = [c for c in CRITERIA if c in criteria]
    options = []
    if metastatic is not None and crpc is not None and met_names:
        options.extend(_later(metastatic, crpc, criteria[c]["row"]) for c in met_names)
    if metastatic is not None and CASTRATION_EXEMPT_CRITERIA & set(met_names):
        for name in CASTRATION_EXEMPT_CRITERIA & set(met_names):
            options.append(_later(metastatic, criteria[name]["row"]))
    is_avpc = bool(options)
    onset = earliest(options)
    fields.update({
        "avpc_criteria": met_names,
        "n_avpc_criteria": len(met_names),
        "is_avpc": is_avpc,
        "avpc_date": onset and onset.get("event_date"),
        "avpc_date_precision": onset and onset.get("date_precision"),
    })
    return fields, sorted(set(rejections))
