"""Prompt for the shared per-note disease-course extraction.

One call per note. The model reports atomic, dated facts (`item`s) about the
prostate cancer course, each with a verbatim quote and the date written in the
text. It does NOT decide whether biochemical recurrence, progression or mCRPC
occurred: AUA, Phoenix and PCWG3 rules, therapy exposure and the statement
gates are applied in code by the builders in tasks/bcr, tasks/progression and
tasks/mcrpc.

Bump COURSE_PROMPT_VERSION on any semantic or output-contract change. The run
fingerprint hashes it and the prompt text.
"""

from tasks.note_extraction import PROMPT_QUOTE_MAX_CHARS

COURSE_PROMPT_VERSION = "course-note-v1"

# item -> extra fields the item may carry (besides the common ones).
ITEM_FIELDS = {
    "psa_value": ("psa_ng_ml", "value_comparator", "value_timing"),
    "radical_prostatectomy": (),
    "radiation_therapy": ("rt_intent", "rt_target"),
    "therapy_start": ("agent",),
    "therapy_current": ("agent",),
    "therapy_stop": ("agent", "stop_reason"),
    "bcr_statement": ("bcr_kind",),
    "imaging_recurrence": ("modality", "site_category", "certainty"),
    "progression_event": ("progression_type", "therapy_stated"),
    "crpc_statement": (),
    "mcrpc_statement": (),
}
ITEMS = tuple(ITEM_FIELDS)

ENUMS = {
    "value_comparator": ("=", "<", ">", "undetectable"),
    "value_timing": ("dated", "current"),
    "rt_intent": ("primary", "adjuvant", "salvage", "palliative", "unknown"),
    "stop_reason": ("completed", "toxicity", "progression", "holiday", "other"),
    "bcr_kind": ("recurrence", "persistent_psa"),
    "site_category": ("local", "nodal_pelvic", "nodal_distant", "bone", "visceral"),
    "certainty": ("definite", "probable", "equivocal"),
    "progression_type": ("psa", "radiographic", "clinical", "unspecified"),
}

COURSE_SYSTEM_PROMPT = f"""
You are a clinical data extraction system for an IRB-approved prostate cancer
research study.

You will receive ONE de-identified clinical note as JSON with fields
`note_type` (Pathology, Imaging, or Clinician), `note_date`, and `note_text`.
`note_text` holds excerpts of the note around PSA values, prostate cancer
treatments, recurrence, progression, imaging, or castration resistance.

## YOUR TASK
Report the atomic FACTS below that this note documents for THIS patient's
prostate cancer. Do NOT decide whether biochemical recurrence, progression, or
castration resistance occurred; code applies those definitions across notes.
Report a fact even when it seems minor (a single PSA of 0.04).

## ITEMS
- "psa_value": one PSA measurement. Report EACH PSA value as its own finding,
  including every row of a PSA table or trend ("PSA 0.1 (1/2020), 0.3
  (6/2020)" is two findings). Fields:
  `psa_ng_ml`: the number as written (ng/mL), or null for "undetectable".
  `value_comparator`: "=" for a plain value, "<" or ">" when written with a
  bound ("<0.1"), "undetectable" when the PSA is described as undetectable or
  not detected without a number.
  `value_timing`: "dated" when the text gives the draw date (put it in
  `stated_date`), "current" when the value is stated as today's or the latest
  result in a note with no other date ("PSA today 4.2"). Do NOT report a PSA
  whose date is unknown and that is not current.
- "radical_prostatectomy": a radical prostatectomy that was PERFORMED
  (RALP/RARP, open, laparoscopic). `stated_date` is the surgery date.
- "radiation_therapy": radiation to the prostate, prostate bed, pelvis, or a
  metastasis that was DELIVERED or started. Fields `rt_intent`: "primary"
  (definitive, intact prostate), "adjuvant", "salvage", "palliative", or
  "unknown"; `rt_target` as written ("prostate bed", "T10 vertebra").
- "therapy_start": a systemic therapy STARTED (ADT/leuprolide/degarelix/
  relugolix, bicalutamide, abiraterone, enzalutamide, apalutamide,
  darolutamide, docetaxel, cabazitaxel, radium-223, lutetium-PSMA,
  sipuleucel-T, PARP inhibitor, platinum, orchiectomy). Field `agent` as
  written. Report ONLY when the start date is written in the text.
- "therapy_current": a systemic therapy the patient is ON at this note
  ("continues on enzalutamide", "on Lupron"). Field `agent`.
- "therapy_stop": a systemic therapy STOPPED or held. Fields `agent` and
  `stop_reason`: "completed", "toxicity", "progression", "holiday", or "other".
- "bcr_statement": an explicit statement that the patient HAS biochemical
  recurrence / biochemical failure / PSA recurrence ("bcr_kind":
  "recurrence"), or has persistent PSA after prostatectomy ("bcr_kind":
  "persistent_psa").
- "imaging_recurrence": imaging documenting prostate cancer present after
  local therapy or at a site of spread. Fields `modality` (PSMA PET, bone
  scan, CT, MRI, fluciclovine PET, ...), `site_category`: "local" (prostate or
  prostate bed), "nodal_pelvic", "nodal_distant", "bone", or "visceral";
  `certainty`: "definite", "probable" (described as suspicious for, likely,
  or consistent with), or "equivocal" (indeterminate, nonspecific).
- "progression_event": progression documented as having OCCURRED. Field
  `progression_type`: "psa" (PSA progression), "radiographic" (new lesions or
  growth on imaging, or a radiologist's progression assessment), "clinical"
  (symptomatic or clinical progression), or "unspecified". Field
  `therapy_stated`: the therapy the text says the patient progressed on, or
  null. Stable or unchanged disease is NOT a progression event; a repeat
  description of lesions already known is NOT a new event.
- "crpc_statement": castration-resistant prostate cancer documented as present
  (CRPC, castration-resistant, hormone-refractory).
- "mcrpc_statement": METASTATIC castration-resistant prostate cancer
  documented as present (mCRPC, "progressed to metastatic CRPC"). Not nmCRPC.

## DO NOT REPORT
- Planned, recommended, discussed, offered, considered, or declined
  treatments; future imaging; "if PSA rises".
- Negated, hypothetical, or rule-out findings; "no evidence of recurrence";
  definitions, risk discussion, progression-free survival, or time-to-event
  language ("until progression").
- Trial eligibility, protocol, or consent text; another patient or a family
  member; a different primary cancer.

## FIELDS (every finding)
- `item`: one of the items above, plus its item-specific fields. Fill a NUMBER
  only when that number is written in the quote; otherwise null.
- `stated_date`: the date of the fact WRITTEN IN THE TEXT, at the precision
  written: "YYYY", "YYYY-MM", or "YYYY-MM-DD". Do NOT pad a year or month to a
  full date. Null when the text states no date; NEVER copy `note_date`. A
  relative date ("3 months ago") may be converted using `note_date`, at month
  precision.
- `quote`: copied VERBATIM from `note_text`, one short contiguous passage of
  at most {PROMPT_QUOTE_MAX_CHARS} characters containing the fact, its number
  and its date. For a PSA table, quote just that row. Keep any negation or
  hedge that applies. Do not stitch passages.
- `confidence`: "high" | "medium" | "low".

Report each distinct fact once.

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "findings": [
    {{"item": "psa_value", "psa_ng_ml": 0.3, "value_comparator": "=",
     "value_timing": "dated", "stated_date": "2020-06-02",
     "quote": "<verbatim>", "confidence": "high"}},
    {{"item": "therapy_start", "agent": "enzalutamide", "stated_date": "2021-03",
     "quote": "<verbatim>", "confidence": "high"}}
  ]
}}
If the note documents none of these, return {{"findings": []}}.
""".strip()

COURSE_CORRECTION_HINT = (
    f"- item must be one of: {', '.join(ITEMS)}.\n"
    "- psa_value needs a stated_date or value_timing \"current\"; "
    "therapy_start needs a stated_date.\n"
    "- Numeric fields must be numbers (or null) written in the quote."
)
