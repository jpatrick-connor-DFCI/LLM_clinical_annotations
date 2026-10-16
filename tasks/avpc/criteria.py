"""Deterministic AVPC rollup: grounded note-level facts -> patient criteria.

Paper numbering (Aparicio et al., Clin Cancer Res 2013, Table 1): C2 =
exclusively visceral metastases, C3 = predominantly lytic bone metastases. The
older binary_subtype / longitudinal_subtype pipelines swap those two.

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

from tasks.longitudinal_subtype.mcrpc_gate import (
    _CASTRATING_THERAPY,
    CASTRATION_EXEMPT_CRITERIA,
    CASTRATION_RESISTANT_CRITERIA,
    METASTATIC_CRITERIA,
    screen_castration_resistant_quote,
    screen_metastatic_quote,
)
from tasks.longitudinal_subtype.met_veto import BOILERPLATE
from tasks.nepc.ne_gate import positive_ihc_markers, screen_asserted, screen_ne_quote
from tasks.onset import date_interval, earliest
from tasks.trial_guard import trial_boilerplate_reason

CRITERIA_VERSION = "avpc-criteria-v2"

CRITERIA = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
BULKY_CM = 5.0
HIGH_GLEASON = 8
LOW_PSA_NG_ML = 10.0
HIGH_VOLUME_BONE_COUNT = 20
ULN_MULTIPLE = 2.0
C5_PAIRING_DAYS = 90
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
    r"\b(?:gleason|gs)\s*(?:score|grade|sum)?\s*(?:of|=|:|is)?\s*(\d{1,2})\b|=\s*(\d{1,2})\b",
    flags=re.IGNORECASE,
)
_GRADE_GROUP = re.compile(r"\b(?:grade\s+group|gg|isup\s+grade)\s*(?:of|:)?\s*([1-5])\b", flags=re.IGNORECASE)
_GRADE_GROUP_GLEASON = {4: 8, 5: 9}

ADT_TERM = re.compile(rf"\b{_CASTRATING_THERAPY}\b", flags=re.IGNORECASE)
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
_MET_WORD = re.compile(
    r"\b(?:metasta\w*|mets|malignan\w*|consistent\s+with|c/w|carcinoma)\b",
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
HIGH_VOLUME = re.compile(
    r"\b(?:innumerable|too\s+numerous\s+to\s+count|tntc|super[\s-]?scan|countless)\b",
    flags=re.IGNORECASE,
)
_AT_LEAST_N = re.compile(
    r"(?:>|≥|>=|\bmore\s+than|\bgreater\s+than|\bover|\bat\s+least)\s*(\d+)",
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


def gleason_in_quote(total, quote):
    if total is None:
        return False
    total = int(total)
    candidates = {int(a) + int(b) for a, b in _GLEASON_SUM.findall(quote or "")}
    for a, b in _GLEASON_TOTAL.findall(quote or ""):
        candidates.add(int(a or b))
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
        if not _MET_WORD.search(quote):
            return "no_metastatic_assertion"
        ok, reason = screen_asserted(quote, VISCERAL_TERM)
        return None if ok else ("no_visceral_term" if reason == "no_term" else reason)
    if item in {"bulky_lymphadenopathy", "bulky_prostate_pelvic_mass"}:
        term = NODE_TERM if item == "bulky_lymphadenopathy" else MASS_TERM
        if item == "bulky_prostate_pelvic_mass" and not PELVIC_SITE.search(quote):
            return "no_prostate_pelvic_site"
        if not size_in_quote(row.get("size_cm"), quote):
            return "size_not_in_quote"
        ok, reason = screen_asserted(quote, term)
        return None if ok else reason
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
    """The component documented last (dates a composite criterion)."""
    dated = [r for r in rows if r is not None and r.get("event_date")]
    if not dated:
        return rows[0]
    return max(dated, key=lambda r: (r["event_date"], r.get("source_note_date") or ""))


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


def _is_high_volume(row):
    quote = row.get("quote") or ""
    count = row.get("count")
    if count is not None and count >= HIGH_VOLUME_BONE_COUNT and number_in_quote(count, quote):
        return True
    if HIGH_VOLUME.search(quote):
        return True
    return any(int(n) >= HIGH_VOLUME_BONE_COUNT for n in _AT_LEAST_N.findall(quote))


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


def evaluate_c3(bone, rejections):
    candidates = []
    for row in bone:
        if row.get("lytic_predominant") is not True:
            continue
        quote = row["quote"]
        if not LYTIC.search(quote):
            _reject(rejections, row, "c3_lytic_not_in_quote")
            continue
        if BLASTIC.search(quote) and not LYTIC_PREDOMINANT.search(quote):
            _reject(rejections, row, "c3_lytic_predominance_not_stated")
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


def evaluate_c5(psa, bone, rejections):
    low = []
    for row in psa:
        value = row.get("psa_ng_ml")
        context = row.get("psa_context")
        if value is None or value > LOW_PSA_NG_ML or context not in PSA_CONTEXT_CUES:
            continue
        if not PSA_CONTEXT_CUES[context].search(row["quote"]):
            _reject(rejections, row, "c5_psa_context_not_in_quote")
            continue
        low.append(row)
    heavy = [r for r in bone if _is_high_volume(r)]
    pairs = []
    for p in low:
        for b in heavy:
            gap = _gap_days(p, b)
            if gap is not None and gap <= C5_PAIRING_DAYS:
                pairs.append((p, b))
    if not pairs:
        return None
    p, b = min(pairs, key=lambda pair: (_later(*pair)["event_date"], _later(*pair).get("source_note_date") or ""))
    return _criterion(
        _later(b, p),
        quote=f"{b['quote']} | {p['quote']}",
        components={"c5_psa_quote": p["quote"], "c5_bone_quote": b["quote"]},
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


def evaluate_c7(rapid, adt, crpc):
    explicit = earliest(rapid)
    candidates = [_criterion(explicit)] if explicit is not None else []
    starts = [
        r for r in adt
        if r.get("date_source") == "stated" and r.get("date_precision") in C7_MIN_PRECISION
    ]
    resist = [r for r in crpc if r.get("date_precision") in C7_MIN_PRECISION]
    start, onset = earliest(starts), earliest(resist)
    if start is None or onset is None:
        return _earliest_criterion(candidates)
    start_iv, onset_iv = _interval(start), _interval(onset)
    # Certainly after the start, and certainly within the window.
    if not onset_iv[0] > start_iv[1]:
        return _earliest_criterion(candidates)
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
    "c6_marker_quote", "c6_lab_quote", "c7_adt_start_quote", "c7_crpc_quote",
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


def _max_date(*dates):
    """The date by which every component held; unknown if any is undated."""
    if any(not d for d in dates):
        return None
    return max(dates)


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
        ("C3", evaluate_c3(get("bone_metastasis"), rejections)),
        ("C4", evaluate_c4(get("bulky_lymphadenopathy"), get("bulky_prostate_pelvic_mass"),
                           get("gleason_high_grade"))),
        ("C5", evaluate_c5(get("low_psa"), get("bone_metastasis"), rejections)),
        ("C6", evaluate_c6(get("ne_marker"), get("elevated_ldh") + get("elevated_cea"),
                           get("malignant_hypercalcemia"))),
        ("C7", evaluate_c7(get("rapid_progression_on_adt"), get("adt_start"),
                           get("castration_resistance"))),
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
        "is_castration_resistant": crpc is not None,
        "crpc_quote": crpc and crpc["quote"],
        "crpc_date": crpc and crpc.get("event_date"),
        "crpc_source": crpc_source,
        "is_mcrpc": metastatic is not None and crpc is not None,
    }
    for name in CRITERIA:
        key = name.lower()
        result = criteria.get(name)
        row = result and result["row"]
        fields[f"{key}_met"] = result is not None
        fields[f"{key}_quote"] = result and result["quote"]
        fields[f"{key}_date"] = row and row.get("event_date")
        fields[f"{key}_source_note_date"] = row and row.get("source_note_date")
    for column in COMPONENT_COLUMNS:
        fields[column] = None
    for result in criteria.values():
        fields.update(result["components"])

    met_names = [c for c in CRITERIA if c in criteria]
    met_date = metastatic and metastatic.get("event_date")
    crpc_date = crpc and crpc.get("event_date")
    options = []
    if metastatic is not None and crpc is not None and met_names:
        first_criterion = min(
            (criteria[c]["row"].get("event_date") for c in met_names
             if criteria[c]["row"].get("event_date")),
            default=None,
        )
        options.append(_max_date(met_date, crpc_date, first_criterion))
    if metastatic is not None and CASTRATION_EXEMPT_CRITERIA & set(met_names):
        for name in CASTRATION_EXEMPT_CRITERIA & set(met_names):
            options.append(_max_date(met_date, criteria[name]["row"].get("event_date")))
    is_avpc = bool(options)
    fields.update({
        "avpc_criteria": met_names,
        "n_avpc_criteria": len(met_names),
        "is_avpc": is_avpc,
        "avpc_date": min((d for d in options if d), default=None),
    })
    return fields, sorted(set(rejections))
