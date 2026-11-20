"""Deterministic AVPC-adjacent rollup: grounded note-level facts -> patient label.

AVPC-adjacent is a study definition, broader than Aparicio et al. 2013, for
patients whose disease looks aggressive-variant without meeting the strict
definition. It has NO disease-state gate: a patient is AVPC-adjacent when any
of these is documented, metastatic / castration-resistant or not:

  A1  any visceral or soft-tissue metastasis: the strict C2 organ list plus
      dura / leptomeninges / spinal cord, bladder / rectum (discrete
      metastases only, not contiguous T4 invasion), and soft-tissue /
      subcutaneous / skin / muscle deposits. Lymph nodes never count, nor does
      the soft-tissue or epidural extension of a bone lesion.
  A2  any lytic bone metastasis, including mixed lytic/blastic.
  A3  modified C5: strict C5's low-PSA component and 90-day pairing, with the
      bone-volume test relaxed to extent wording (extensive, widespread,
      diffuse, numerous, multiple, multifocal, high-volume, heavy burden).
  A4  an explicit "aggressive variant" / AVPC / anaplastic / variant-CRPC
      statement.
  C1-C7  any strict Aparicio criterion, ungated (computed by
      tasks/avpc/criteria.py from this pipeline's own findings).

As in tasks/avpc, the model never decides a criterion: every finding is
screened here, so `--rebuild-labels-only` re-applies a change without LLM
calls. Bump ADJ_CRITERIA_VERSION on any change here.
"""

import re

from tasks.avpc import criteria as strict
from tasks.avpc.criteria import (
    BLASTIC,
    BONE_TERM,
    LYTIC,
    _criterion,
    _is_high_volume,
    evaluate_c5,
    metastatic_assertion_reason,
)
from tasks.nepc.ne_gate import screen_asserted
from tasks.onset import earliest
from tasks.trial_guard import trial_boilerplate_reason

ADJ_CRITERIA_VERSION = "avpc-adjacent-criteria-v5"

ADJ_CRITERIA = ("A1", "A2", "A3", "A4")
ADJ_KEYS = {"A1": "a1_visceral", "A2": "a2_lytic", "A3": "a3_c5mod", "A4": "a4_statement"}
AV_STATEMENT = "aggressive_variant_statement"

# --- A1: extended visceral / soft-tissue sites ------------------------------------

# Bladder and rectum are listed apart: only a metastasis word qualifies them
# (a "bladder mass" is usually the primary growing into it).
_ORGAN = (
    r"liver|hepatic|lungs?|pulmonary|adrenals?|brain|cerebral|cerebell\w*|intracranial|"
    r"dura|dural|leptomening\w*|peritone\w*|pleura\w*|pancrea\w*|splen\w*|spleen|"
    r"renal|kidneys?|omentum|omental|mesenter\w*|visceral|"
    r"soft[\s-]+tissue|subcutaneous|cutaneous|skin|dermal|muscle|muscular|intramuscular|"
    r"chest[\s-]+wall|abdominal[\s-]+wall"
)
_HOLLOW = r"(?:urinary\s+)?bladder|vesical|rect(?:um|al)"
_LESION = (
    r"metasta\w*|mets|lesions?|mass(?:es)?|nodules?|disease|involvement|deposits?|"
    r"implants?|carcinomatosis"
)
_MET_ONLY = r"metasta\w*|mets|deposits?|implants?"
EXTENDED_VISCERAL_TERM = re.compile(
    rf"\b(?:(?:{_ORGAN})(?:[\s-]+based)?\s+(?:{_LESION})|"
    rf"(?:{_HOLLOW})\s+(?:{_MET_ONLY})|"
    r"spinal\s+cord\s+(?:metasta\w*|mets|lesions?|deposits?)|intramedullary\s+(?:metasta\w*|mets|lesions?)|"
    rf"(?:metasta\w*|mets)\s+(?:to|in|involving|within)\s+(?:the\s+)?(?:{_ORGAN}|{_HOLLOW}|spinal\s+cord))\b",
    flags=re.IGNORECASE,
)
_EXTENSION_SITE = (
    r"soft[\s-]+tissue|subcutaneous|cutaneous|skin|dermal|muscle|muscular|intramuscular|"
    r"dura|dural|leptomening\w*|spinal\s+cord|intramedullary|"
    r"chest[\s-]+wall|abdominal[\s-]+wall"
)
SOFT_TISSUE_SITE = re.compile(rf"\b(?:{_EXTENSION_SITE})\b", flags=re.IGNORECASE)
HOLLOW_SITE = re.compile(rf"\b(?:{_HOLLOW})\b", flags=re.IGNORECASE)
_ADJ_MET_WORD = re.compile(
    r"\b(?:metasta\w*|mets|malignan\w*|carcinoma|"
    r"carcinomatosis|deposits?|implants?)\b",
    flags=re.IGNORECASE,
)
# A soft-tissue component of a bone lesion is bone disease, not a deposit.
# Vertebral levels ("L3", "T12") name a bone without a bone word.
_VERTEBRAL_LEVEL = re.compile(r"\b[CTLS]\d{1,2}\b", flags=re.IGNORECASE)
_BONE_SOURCE = rf"(?:{BONE_TERM.pattern}|{_VERTEBRAL_LEVEL.pattern})"
BONE_EXTENSION = re.compile(
    r"\b(?:(?:associated|accompanying|adjacent|extra-?osseous|epidural|paraspinal|"
    rf"paravertebral)\s+(?:\w+\s+){{0,2}}(?:{_EXTENSION_SITE})|"
    rf"(?:{_EXTENSION_SITE})\s+(?:component|extension|expansion)|"
    r"(?:arising|extending|extends?|extension)\s+(?:from|out\s+of)\s+(?:the\s+)?(?:\w+\s+){0,3}"
    rf"(?:{_BONE_SOURCE}|lesion)|"
    rf"{_BONE_SOURCE}\s+(?:\w+\s+){{0,4}}?"
    r"(?:extend\w*\s+into|extension\s+into|with\s+(?:direct|contiguous))\s+"
    rf"(?:the\s+)?(?:{_EXTENSION_SITE}))\b",
    flags=re.IGNORECASE,
)
_SITE_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _site_context(quote, hit):
    """Keep linked clauses, but do not borrow invasion from another sentence."""
    before = list(_SITE_SENTENCE_END.finditer(quote, 0, hit.start()))
    after = _SITE_SENTENCE_END.search(quote, hit.end())
    return quote[before[-1].end() if before else 0:after.start() if after else len(quote)]


def _names_bone(quote):
    return bool(BONE_TERM.search(quote) or LYTIC.search(quote) or BLASTIC.search(quote)
                or _VERTEBRAL_LEVEL.search(quote))


LOCAL_INVASION = re.compile(
    r"\b(?:invad\w*|invasion|infiltrat\w*|direct(?:ly)?\s+extend\w*|direct\s+extension|"
    r"extends?\s+into|extension\s+into|contiguous|abut\w*|t4)\b",
    flags=re.IGNORECASE,
)

# --- A2 / A3 ----------------------------------------------------------------------

EXTENT_TERM = re.compile(
    r"\b(?:extensive(?:ly)?|widespread|diffuse(?:ly)?|numerous|multiple|multifocal|"
    r"high[\s-]+volume|heavy)\s+"
    r"(?:(?:metastatic|prostate|cancer|disease|tumou?r|burden|of|the|in|throughout|"
    r"mixed|axial|appendicular|new|additional|progressive|stable|known)\s+){0,3}"
    r"(?:bone|bony|osseous|skeletal|osteoblastic|osteolytic|sclerotic|lytic|blastic)\b",
    flags=re.IGNORECASE,
)

# --- A4 ---------------------------------------------------------------------------

AV_TERM = re.compile(
    r"\b(?:aggressive[\s-]+variant|avpc|anaplastic|"
    r"variant\s+(?:n?m?crpc|castrat\w*[\s-]+resistan\w*))\b",
    flags=re.IGNORECASE,
)
# Leading and trailing question marks are chart hedges. Do not cross sentence
# punctuation to borrow a question about treatment from an asserted diagnosis.
AV_QUESTION = re.compile(
    r"\?\s*(?:of\s+|for\s+)?(?:aggressive|avpc|anaplastic|variant)|"
    rf"(?:{AV_TERM.pattern})(?:\s+[\w-]+){{0,5}}\s*\(?\s*\?",
    flags=re.IGNORECASE,
)
ANAPLASTIC_OTHER = re.compile(
    r"\banaplastic\s+(?:\w+\s+){0,3}?(?:thyroid|lymphoma|kinase|astrocytoma|glioma|"
    r"ependymoma|meningioma|oligodendroglioma|wilms|seminoma|large[\s-]+cell|sarcoma)\b|"
    r"\b(?:alcl|alk)\b",
    flags=re.IGNORECASE,
)


# --- Per-finding screens ------------------------------------------------------------


def _screen_extended_visceral(quote):
    reason = metastatic_assertion_reason(quote, _ADJ_MET_WORD)
    if reason:
        return reason
    ok, reason = screen_asserted(quote, EXTENDED_VISCERAL_TERM)
    if not ok:
        return "no_visceral_term" if reason == "no_term" else reason
    sites = list(EXTENDED_VISCERAL_TERM.finditer(quote))
    remaining = []
    excluded = []
    for hit in sites:
        text = hit.group(0)
        context = _site_context(quote, hit)
        if SOFT_TISSUE_SITE.search(text) and _names_bone(context):
            if screen_asserted(context, BONE_EXTENSION)[0]:
                excluded.append("bone_extension_not_visceral")
                continue
        if HOLLOW_SITE.search(text) and screen_asserted(context, LOCAL_INVASION)[0]:
            excluded.append("local_invasion_not_metastasis")
            continue
        remaining.append(hit)
    if not remaining:
        return excluded[0] if excluded else "no_visceral_term"
    return None


def _screen_av_statement(quote):
    other = ANAPLASTIC_OTHER.search(quote)
    if other is not None:
        return f"non_prostate_anaplastic:{other.group(0).lower()}"
    if AV_QUESTION.search(quote):
        return "negated_or_hedged:question"
    ok, reason = screen_asserted(quote, AV_TERM)
    if not ok:
        return "no_av_term" if reason == "no_term" else reason
    return None


def screen_finding(row):
    """Return None when the finding can count toward an adjacent criterion,
    else a rejection reason. Items other than visceral metastases and AV
    statements use the strict AVPC screen unchanged."""
    item = row.get("item")
    quote = row.get("quote") or ""
    if item not in {"visceral_metastasis", AV_STATEMENT}:
        return strict.screen_finding(row)
    trial_reason = row.get("_trial_context_reason") or trial_boilerplate_reason(quote)
    if trial_reason:
        return trial_reason
    if item == "visceral_metastasis":
        return _screen_extended_visceral(quote)
    return _screen_av_statement(quote)


# --- Criteria -----------------------------------------------------------------------


def _first(rows):
    first = earliest(rows)
    return _criterion(first) if first is not None else None


def evaluate_a1(visceral):
    return _first(visceral)


def evaluate_a2(bone, rejections):
    candidates = []
    for row in bone:
        if row.get("lytic_component") is not True and row.get("lytic_predominant") is not True:
            continue
        ok, reason = screen_asserted(row["quote"], LYTIC)
        if not ok:
            rejections.append(
                f"bone_metastasis:a2_{'lytic_not_in_quote' if reason == 'no_term' else reason}"
            )
            continue
        candidates.append(row)
    return _first(candidates)


def is_high_volume_extended(row):
    """The paper's volume test, or asserted extent wording in the quote."""
    if _is_high_volume(row):
        return True
    quote = row.get("quote") or ""
    if not EXTENT_TERM.search(quote):
        return False
    ok, _ = screen_asserted(quote, EXTENT_TERM)
    return ok


def evaluate_a3(psa, bone, rejections, adt=()):
    result = evaluate_c5(psa, bone, rejections, is_high_volume=is_high_volume_extended, adt=adt)
    if result is None:
        return None
    components = result["components"]
    result["components"] = {
        "a3_psa_quote": components["c5_psa_quote"],
        "a3_bone_quote": components["c5_bone_quote"],
    }
    return result


def evaluate_a4(statements):
    return _first(statements)


# --- Patient --------------------------------------------------------------------------

COMPONENT_COLUMNS = ("a3_psa_quote", "a3_bone_quote")
# Strict fields carried through as audit / ungated criteria.
STRICT_STATE_FIELDS = (
    "is_metastatic", "metastatic_date", "is_castration_resistant", "crpc_date", "is_mcrpc",
)


def evaluate_patient(rows):
    """Return (label fields, rejection reasons) for one patient's grounded findings.

    Field names are unprefixed; the runner prefixes them for the label file.
    """
    rejections = []
    by_item = {}
    for row in rows:
        reason = screen_finding(row)
        if reason:
            rejections.append(f"{row.get('item')}:{reason}")
            continue
        by_item.setdefault(row["item"], []).append(row)
    get = lambda item: by_item.get(item, [])  # noqa: E731

    criteria = {}
    for name, result in (
        ("A1", evaluate_a1(get("visceral_metastasis"))),
        ("A2", evaluate_a2(get("bone_metastasis"), rejections)),
        ("A3", evaluate_a3(get("low_psa"), get("bone_metastasis"), rejections, get("adt_start"))),
        ("A4", evaluate_a4(get(AV_STATEMENT))),
    ):
        if result is not None:
            criteria[name] = result

    # Strict C1-C7 from the same findings, ignoring the mCRPC gate. The AV
    # statement is not a strict item, so it is left out of the strict pass.
    strict_fields, strict_rejections = strict.evaluate_patient(
        [r for r in rows if r.get("item") != AV_STATEMENT]
    )
    seen = set(rejections)
    rejections += [f"strict:{r}" for r in strict_rejections if r not in seen]

    fields = {name: strict_fields[name] for name in STRICT_STATE_FIELDS}
    fields["is_avpc_strict"] = strict_fields["is_avpc"]
    for name in ADJ_CRITERIA:
        key = ADJ_KEYS[name]
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

    met = [name for name in ADJ_CRITERIA if name in criteria]
    onsets = [criteria[name]["row"] for name in met]
    for name in strict.CRITERIA:
        key = name.lower()
        for suffix in ("met", "quote", "date", "date_precision", "source_note_date"):
            fields[f"{key}_{suffix}"] = strict_fields[f"{key}_{suffix}"]
        if strict_fields[f"{key}_met"]:
            met.append(name)
            onsets.append({"event_date": strict_fields[f"{key}_date"],
                           "date_precision": strict_fields[f"{key}_date_precision"],
                           "source_note_date": strict_fields[f"{key}_source_note_date"]})

    onset = earliest(onsets)
    fields.update({
        "criteria": met,
        "n_criteria": len(met),
        "is_avpc_adjacent": bool(met),
        "avpc_adjacent_date": onset and onset.get("event_date"),
        "avpc_adjacent_date_precision": onset and onset.get("date_precision"),
    })
    return fields, sorted(set(rejections))
