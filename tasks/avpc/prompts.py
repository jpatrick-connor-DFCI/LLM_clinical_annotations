"""Prompt for the per-note AVPC extraction task.

One call per note. The model reports atomic facts (`item`s), each with a
verbatim quote and the date written in the text. It does NOT decide whether a
criterion is met: thresholds, pairings, exclusivity, and timing are applied in
code by tasks/avpc/criteria.py, which also re-checks every number against the
quote.

Criterion numbering follows the PAPER (Aparicio et al., Clin Cancer Res 2013,
Table 1): C2 = exclusively visceral metastases, C3 = predominantly lytic bone
metastases. The older binary_subtype / longitudinal_subtype pipelines use the
reverse order for those two.

Bump AVPC_PROMPT_VERSION on any semantic or output-contract change. The run
fingerprint hashes it and the prompt text.
"""

from tasks.note_extraction import PROMPT_QUOTE_MAX_CHARS

AVPC_PROMPT_VERSION = "avpc-note-v1"

# item -> extra fields the item may carry (besides the common ones).
ITEM_FIELDS = {
    "metastatic_disease": ("sites",),
    "castration_resistance": (),
    "adt_start": ("agent",),
    "rapid_progression_on_adt": (),
    "small_cell_histology": ("specimen_site",),
    "visceral_metastasis": ("organ", "exclusively_visceral"),
    "bone_metastasis": ("lytic_predominant", "count", "high_volume_descriptor"),
    "bulky_lymphadenopathy": ("size_cm",),
    "bulky_prostate_pelvic_mass": ("size_cm",),
    "gleason_high_grade": ("gleason_total",),
    "low_psa": ("psa_ng_ml", "psa_context"),
    "ne_marker": ("marker", "specimen"),
    "elevated_ldh": ("value", "uln"),
    "elevated_cea": ("value", "uln"),
    "malignant_hypercalcemia": (),
}
ITEMS = tuple(ITEM_FIELDS)
PSA_CONTEXTS = ("initial_presentation_pre_adt", "symptomatic_crpc_progression")
NE_MARKER_SPECIMENS = ("tissue", "serum")

AVPC_CRITERIA_TEXT = """
### Aggressive-variant prostate cancer (Aparicio et al. 2013, Table 1)
AVPC is metastatic, castration-resistant prostate carcinoma with at least one of:
C1  Histologic evidence of small-cell (pure or mixed) prostate carcinoma.
C2  Exclusively visceral metastases (no bone or nodal metastases).
C3  Radiographically predominantly lytic bone metastases.
C4  Bulky (>=5 cm) lymphadenopathy, or a bulky (>=5 cm) high-grade (Gleason
    >=8) tumor mass in the prostate/pelvis.
C5  Low PSA (<=10 ng/mL) at initial presentation (before ADT) or at
    symptomatic progression in the castrate state, PLUS high-volume (>=20)
    bone metastases.
C6  Neuroendocrine markers on histology (chromogranin A or synaptophysin
    staining) or in serum (abnormally elevated chromogranin A or GRP), PLUS
    elevated serum LDH (>=2x the upper limit of normal), malignant
    hypercalcemia, or elevated serum CEA (>=2x the upper limit of normal).
C7  Short interval (<=6 months) to androgen-independent progression after
    initiation of hormonal therapy.
Small-cell carcinoma (C1) need not be castration-resistant.
""".strip()

AVPC_SYSTEM_PROMPT = f"""
You are a clinical data extraction system for an IRB-approved prostate cancer
research study.

You will receive ONE de-identified clinical note as JSON with fields
`note_type` (Pathology, Imaging, or Clinician), `note_date`, and `note_text`.
`note_text` holds excerpts of the note around words related to metastases,
castration resistance, or the criteria below.

{AVPC_CRITERIA_TEXT}

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
- "visceral_metastasis": a visceral (liver, lung, adrenal, brain, peritoneal,
  pleural, other organ) metastasis. Field `organ`. Field
  `exclusively_visceral`: true ONLY when this note states there are NO bone and
  NO nodal metastases, otherwise false.
- "bone_metastasis": bone metastases. Field `lytic_predominant`: true only if
  the lesions are described as lytic / osteolytic predominantly, false if
  blastic, sclerotic, or mixed without lytic predominance, null if not
  described. Field `count`: number of bone lesions if a number is written.
  Field `high_volume_descriptor`: the words used for extent ("innumerable",
  "too numerous to count", "superscan", "diffuse", "extensive"), or null.
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

## DO NOT REPORT
- Negated, hedged, suspected, possible, "concerning for", rule-out,
  differential, or pending findings; "no evidence of metastatic disease".
- Risk, surveillance, or future discussion; trial eligibility criteria,
  protocol, or consent text; another patient or a family member.
- Findings of a different primary cancer.
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
    {{"item": "bone_metastasis", "lytic_predominant": true, "count": null,
     "high_volume_descriptor": "innumerable", "stated_date": null,
     "quote": "<verbatim>", "confidence": "high"}},
    {{"item": "low_psa", "psa_ng_ml": 2.1,
     "psa_context": "symptomatic_crpc_progression", "stated_date": "2021-03",
     "quote": "<verbatim>", "confidence": "medium"}}
  ]
}}
If the note documents none of these, return {{"findings": []}}.
""".strip()

AVPC_CORRECTION_HINT = (
    f"- item must be one of: {', '.join(ITEMS)}.\n"
    "- Numeric fields must be numbers (or null) written in the quote."
)
