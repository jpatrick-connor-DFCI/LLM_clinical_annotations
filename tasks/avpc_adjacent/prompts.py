"""Prompt for the per-note AVPC-adjacent extraction task.

One call per note, the same contract as tasks/avpc: the model reports atomic
facts (`item`s), each with a verbatim quote and the date written in the text,
and never decides whether a criterion is met. tasks/avpc_adjacent/criteria.py
applies the strict Aparicio criteria (ungated) and the four adjacent criteria
in code.

The item set is AVPC's (so strict C1-C7 can be recomputed from this run's own
findings) with three changes:
  - visceral_metastasis covers an extended site list and says what does not
    count (lymph nodes, bone-lesion extension, contiguous T4 invasion);
  - bone_metastasis gains `lytic_component` (any lytic, including mixed);
  - a new aggressive_variant_statement item.

This text is a deliberate copy of tasks/avpc/prompts.py, not an import of it:
editing the AVPC prompt would change the AVPC run fingerprint.

Criterion numbering follows the PAPER (C2 = exclusively visceral, C3 =
predominantly lytic), as in tasks/avpc.

Bump AVPC_ADJACENT_PROMPT_VERSION on any semantic or output-contract change.
"""

from tasks.avpc.prompts import AVPC_CRITERIA_TEXT, NE_MARKER_SPECIMENS, PSA_CONTEXTS
from tasks.note_extraction import PROMPT_QUOTE_MAX_CHARS

AVPC_ADJACENT_PROMPT_VERSION = "avpc-adjacent-note-v1"

# item -> extra fields the item may carry (besides the common ones).
ITEM_FIELDS = {
    "metastatic_disease": ("sites",),
    "castration_resistance": (),
    "adt_start": ("agent",),
    "rapid_progression_on_adt": (),
    "small_cell_histology": ("specimen_site",),
    "visceral_metastasis": ("organ", "exclusively_visceral"),
    "bone_metastasis": ("lytic_component", "lytic_predominant", "count", "high_volume_descriptor"),
    "bulky_lymphadenopathy": ("size_cm",),
    "bulky_prostate_pelvic_mass": ("size_cm",),
    "gleason_high_grade": ("gleason_total",),
    "low_psa": ("psa_ng_ml", "psa_context"),
    "ne_marker": ("marker", "specimen"),
    "elevated_ldh": ("value", "uln"),
    "elevated_cea": ("value", "uln"),
    "malignant_hypercalcemia": (),
    "aggressive_variant_statement": ("term",),
}
ITEMS = tuple(ITEM_FIELDS)
# Re-exported for the runner's enum validation (same values as tasks/avpc).
ENUMS = {"psa_context": PSA_CONTEXTS, "specimen": NE_MARKER_SPECIMENS}

ADJACENT_CRITERIA_TEXT = """
### AVPC-adjacent features (study definition, broader than Aparicio)
These are recorded for every patient, whether or not the disease is metastatic
or castration-resistant:
A1  Any visceral or soft-tissue metastasis (see `visceral_metastasis`).
A2  Any lytic bone metastasis, including mixed lytic/blastic lesions.
A3  Low PSA (as in C5) PLUS bone metastases described as extensive,
    widespread, diffuse, numerous, multiple, multifocal, high-volume, or a
    heavy bone burden (no lesion count needed).
A4  An explicit statement that this patient has aggressive-variant prostate
    cancer (AVPC), anaplastic prostate cancer, or variant CRPC.
""".strip()

AVPC_ADJACENT_SYSTEM_PROMPT = f"""
You are a clinical data extraction system for an IRB-approved prostate cancer
research study.

You will receive ONE de-identified clinical note as JSON with fields
`note_type` (Pathology, Imaging, or Clinician), `note_date`, and `note_text`.
`note_text` holds excerpts of the note around words related to metastases,
castration resistance, or the criteria below.

{AVPC_CRITERIA_TEXT}

{ADJACENT_CRITERIA_TEXT}

## YOUR TASK
Do NOT decide whether a criterion is met. Report the atomic FACTS below that
this note documents as PRESENT for this patient's prostate cancer. Code
combines facts across notes and applies every threshold, so report a fact even
when it does not reach a threshold on its own (e.g. a 3.2 cm node, an LDH
1.4x ULN), as long as the item's definition fits.

## ITEMS
- "metastatic_disease": distant (M1) metastatic prostate cancer documented as
  present: bone, visceral, or distant nodal metastases, or an explicit
  "metastatic", mCRPC, mHSPC, or M1 statement. Regional pelvic nodes alone
  (N1, "2 of 14 pelvic lymph nodes") do NOT qualify. Field `sites`: list of
  sites as written, or null.
- "castration_resistance": castration-resistant disease documented as present:
  CRPC, castration-resistant, hormone-refractory, androgen-independent, or PSA
  or radiographic progression stated to occur ON ADT or another castrating
  therapy.
- "adt_start": initiation of androgen deprivation (LHRH agonist/antagonist,
  orchiectomy, or first hormonal therapy). Field `agent` as written. The
  `stated_date` is the START date.
- "rapid_progression_on_adt": an explicit statement that the disease became
  castration-resistant or progressed within about 6 months of starting
  hormonal therapy ("progressed within 4 months of ADT", "primary
  refractory to ADT").
- "small_cell_histology": small-cell (pure or mixed) carcinoma of the
  prostate on histology. Field `specimen_site`.
- "visceral_metastasis": a metastasis of prostate cancer to a visceral organ
  or soft tissue: liver, lung, adrenal, brain, dura / leptomeninges / spinal
  cord, peritoneum, pleura, pancreas, spleen, kidney, omentum, mesentery,
  bladder, rectum, or a soft-tissue, subcutaneous, skin, or muscle deposit.
  Field `organ`. Field `exclusively_visceral`: true ONLY when this note states
  there are NO bone and NO nodal metastases, otherwise false.
  Do NOT report as visceral: lymph nodes of any station; the soft-tissue
  component, extraosseous extension, or epidural extension of a bone lesion
  (report that as bone_metastasis); direct local invasion of the bladder,
  rectum, or seminal vesicles by the prostate tumor (only a separate
  metastatic deposit in those organs counts).
- "bone_metastasis": bone metastases. Field `lytic_component`: true if any
  lesion is described as lytic / osteolytic, including mixed lytic and
  blastic; false if described only as blastic or sclerotic; null if not
  described. Field `lytic_predominant`: true only if the lesions are described
  as lytic / osteolytic predominantly, false if blastic, sclerotic, or mixed
  without lytic predominance, null if not described. Field `count`: number of
  bone lesions if a number is written. Field `high_volume_descriptor`: the
  words used for extent ("innumerable", "too numerous to count", "superscan",
  "extensive", "widespread", "diffuse", "numerous", "multiple", "multifocal",
  "high-volume", "heavy burden"), or null.
- "bulky_lymphadenopathy": an enlarged / metastatic lymph node or nodal mass
  with a measurement. Field `size_cm`: its largest dimension in cm (convert
  mm).
- "bulky_prostate_pelvic_mass": a tumor mass in the prostate or pelvis with a
  measurement. Field `size_cm` as above.
- "gleason_high_grade": a Gleason score for the prostate cancer. Field
  `gleason_total` (e.g. 9 for 4+5).
- "low_psa": a PSA value. Field `psa_ng_ml`. Field `psa_context`:
  "initial_presentation_pre_adt" if this PSA was at initial diagnosis before
  any hormonal therapy, "symptomatic_crpc_progression" if it was at
  symptomatic progression while on ADT / castrate, else null.
- "ne_marker": a POSITIVE neuroendocrine marker: chromogranin A or
  synaptophysin staining on tissue, or an abnormally elevated serum
  chromogranin A or GRP. Field `marker`. Field `specimen`: "tissue" or
  "serum".
- "elevated_ldh": a serum LDH value. Fields `value` and `uln` (the upper limit
  of normal as written, or null).
- "elevated_cea": a serum CEA value. Fields `value` and `uln`.
- "malignant_hypercalcemia": hypercalcemia attributed to malignancy.
- "aggressive_variant_statement": the note states that THIS patient's
  prostate cancer is aggressive-variant prostate cancer (AVPC,
  "aggressive variant"), anaplastic prostate cancer, or variant CRPC. Field
  `term`: the words used. Generic "aggressive disease" or "high-risk" does NOT
  qualify, nor does androgen-indifferent wording on its own.

## DO NOT REPORT
- Negated, hedged, suspected, possible, "concerning for", "?AVPC", rule-out,
  differential, or pending findings; "no evidence of metastatic disease".
- Risk, surveillance, or future discussion; trial eligibility criteria,
  protocol, or consent text; another patient or a family member.
- Findings of a different primary cancer (e.g. anaplastic thyroid carcinoma,
  anaplastic large cell lymphoma).
- Treatment alone as evidence of a state (docetaxel, abiraterone,
  enzalutamide, lutetium without a stated progression or metastatic site).

## FIELDS (every finding)
- `item`: one of the items above.
- Its item-specific fields. Fill a NUMBER only when that number is written in
  the quote; otherwise null. Never estimate or compute a number from text
  that is not in the quote.
- `stated_date`: the date of the fact WRITTEN IN THE TEXT, at the precision
  written: "YYYY", "YYYY-MM", or "YYYY-MM-DD". Do NOT pad a year or month to a
  full date. Null when the text states no date; NEVER copy `note_date`. A
  relative date ("3 months ago") may be converted using `note_date`, at month
  precision.
- `quote`: copied VERBATIM from `note_text`, one contiguous passage of at most
  {PROMPT_QUOTE_MAX_CHARS} characters, containing the fact and any number you
  report. Keep any negation or hedge that applies. Do not stitch passages.
- `confidence`: "high" | "medium" | "low".

Report each distinct fact once. Report every distinct dated value (e.g. each
PSA or LDH with its own date) as its own finding.

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "findings": [
    {{"item": "bone_metastasis", "lytic_component": true,
     "lytic_predominant": false, "count": null,
     "high_volume_descriptor": "extensive", "stated_date": null,
     "quote": "<verbatim>", "confidence": "high"}},
    {{"item": "visceral_metastasis", "organ": "liver",
     "exclusively_visceral": false, "stated_date": "2022-04",
     "quote": "<verbatim>", "confidence": "high"}},
    {{"item": "aggressive_variant_statement", "term": "aggressive variant",
     "stated_date": null, "quote": "<verbatim>", "confidence": "medium"}}
  ]
}}
If the note documents none of these, return {{"findings": []}}.
""".strip()

AVPC_ADJACENT_CORRECTION_HINT = (
    f"- item must be one of: {', '.join(ITEMS)}.\n"
    "- Numeric fields must be numbers (or null) written in the quote."
)
