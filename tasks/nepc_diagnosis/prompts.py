"""Prompts for the strict-precision NEPC diagnosis-date task.

Two stages: a per-chunk map that proposes candidate diagnostic statements, and a
per-patient adjudication that returns one final yes/no plus the earliest
diagnosis date.

Bump PROMPT_SCHEMA_VERSION on any semantic or output-contract change; the
stage-2 run fingerprint hashes this value and both prompt texts, so a changed
prompt forces --overwrite rather than mixing output generations.
"""

PROMPT_SCHEMA_VERSION = "nepc-dx-v2"


QUALIFYING_DEFINITION = """
### What qualifies as an NEPC diagnosis
A qualifying event is EITHER of these, and nothing else:
  (a) An explicitly stated diagnosis of small-cell carcinoma or neuroendocrine
      carcinoma of the prostate (including "NEPC", "t-NEPC", "small cell
      prostate cancer", "oat cell carcinoma"), asserted as an established
      diagnosis.
  (b) A documented histologic transformation of the patient's prostate
      adenocarcinoma to neuroendocrine or small-cell carcinoma, asserted as
      having occurred.

### What does NOT qualify (report nothing for these)
- Negated statements: "no small cell component", "negative for neuroendocrine
  carcinoma", "small cell carcinoma is excluded".
- Immunohistochemistry results alone: "positive for synaptophysin",
  "chromogranin positive", "CD56 reactive", "INSM1 positive" WITHOUT a stated
  diagnosis of neuroendocrine or small-cell carcinoma in the same statement.
- "Neuroendocrine features", "neuroendocrine differentiation", or "focal
  neuroendocrine differentiation" without a stated carcinoma diagnosis.
- Hedged, suspected, possible, probable, favored, concerning-for, equivocal,
  rule-out, differential-diagnosis, or planned/pending-workup wording.
- An isolated mention of the term with no diagnostic assertion attached.
- A neuroendocrine or small-cell carcinoma of a DIFFERENT primary site (lung,
  bladder, GI, unknown primary), or one in the family history.
- Discussion of NEPC as a general risk, a possible future event, a clinical
  trial topic, or educational text.
- Stock/boilerplate text describing a study POPULATION rather than this
  patient: clinical trial eligibility criteria ("inclusion criteria: patients
  with small cell carcinoma are eligible", "exclusion criteria: history of
  neuroendocrine carcinoma"), protocol titles and study descriptions ("a Phase
  II study of ... in neuroendocrine prostate cancer"), consent forms, cohort
  definitions, and registry or questionnaire templates. Such text names NEPC
  because the trial targets NEPC, not because the patient has it. A patient
  being screened for, consented to, or enrolled on an NEPC trial is NOT
  evidence of an NEPC diagnosis -- report the underlying pathology diagnosis if
  one is separately stated, and otherwise report nothing.

Only a pathology diagnosis line or an oncology clinician's stated, established
diagnosis qualifies. When in doubt, report nothing.
""".strip()


NEPC_DX_MAP_PROMPT = f"""
You are the candidate-identification stage of a clinical data extraction system
for an IRB-approved prostate cancer research study. You will receive one chunk
of a single patient's de-identified note snippets.

{QUALIFYING_DEFINITION}

## TASK
Return every statement in this chunk that is a candidate qualifying NEPC
diagnosis, with a verbatim quote. Report each distinct statement once; do not
emit one entry per repetition of the same copy-forward sentence.

A later adjudication stage reviews all candidates for this patient, so report a
candidate whenever a statement plausibly meets (a) or (b), and record honestly
in `assertion_type` and `confidence` how firmly it is asserted. Do NOT report
anything matching the "does NOT qualify" list above.

## RULES
- Use only the supplied snippets. Never infer a diagnosis from stains, markers,
  treatment choice (for example platinum chemotherapy), or clinical suspicion.
- The diagnosis must concern the patient's own prostate-derived disease.
- `quote` must be copied verbatim from one supplied snippet, and must itself
  contain both the diagnosis term and the words that assert it. Do not stitch
  together text from two places. Keep the quote to one or two sentences.
- `source_note_date` must exactly copy the `note_date` of the snippet the quote
  came from.
- `stated_diagnosis_date` is a diagnosis date explicitly stated in the text
  (for example "small cell carcinoma diagnosed in March 2021"). Return null when
  no date is stated in the text; NEVER copy the note date into this field.
  Dates must be valid `YYYY-MM-DD`: normalize a stated year to January 1 and a
  stated year-month to the first of that month. Never emit `xx`, `00`, seasons,
  ranges, or impossible calendar dates.
- `evidence_type`: "stated_diagnosis" | "histologic_transformation".
- `assertion_type`: "established" (asserted as the patient's diagnosis) |
  "reported_history" (attributed to an outside record or a prior note).
- `modality`: "pathology" | "clinical".
- `confidence`: "high" | "medium" | "low".

## OUTPUT
Return only valid JSON:
{{
  "candidates": [
    {{
      "evidence_type": "stated_diagnosis",
      "assertion_type": "established",
      "stated_diagnosis_date": "2021-03-01",
      "source_note_date": "2021-04-12",
      "modality": "pathology",
      "quote": "<verbatim>",
      "confidence": "high"
    }}
  ]
}}
Return {{"candidates": []}} when the chunk contains no qualifying statement.
That is the expected answer for most chunks.
""".strip()


NEPC_DX_SYNTHESIS_PROMPT = f"""
You are the adjudication stage of a clinical data extraction system. You will
receive every candidate NEPC diagnosis statement extracted from one prostate
cancer patient's notes, across all evidence chunks.

{QUALIFYING_DEFINITION}

## TASK
Decide ONCE for this patient whether the record establishes a qualifying NEPC
diagnosis, and if so, identify the EARLIEST qualifying statement.

Adjudicate against the definition above, not against the candidate count. Many
weak candidates do not add up to one qualifying diagnosis; a single unambiguous
pathology diagnosis line is sufficient. Reject the patient when every candidate
is negated, hedged, marker-only, "features/differentiation" only, attributable
to a non-prostate primary, or an isolated mention without a diagnostic
assertion.

## RULES
- Use only the supplied candidates. Never introduce a quote, a date, or a fact
  that is not present in them.
- `supporting_quote` and `source_note_date` must be copied EXACTLY, and as a
  matched pair, from the single candidate you select.
- Select the candidate that is the earliest qualifying evidence of the
  diagnosis, ranking by stated diagnosis date when one exists and otherwise by
  source note date.
- `diagnosis_date` is the diagnosis date stated in that candidate's text.
  Return null when the selected candidate states no date -- a downstream step
  falls back to the note date and records that it did so. Do not guess.
- When `has_nepc_diagnosis` is false, set every other field to null and give a
  one-sentence `rationale` naming the disqualifying reason.

## OUTPUT
Return only valid JSON:
{{
  "has_nepc_diagnosis": true,
  "evidence_type": "stated_diagnosis",
  "assertion_type": "established",
  "diagnosis_date": "2021-03-01",
  "source_note_date": "2021-04-12",
  "modality": "pathology",
  "supporting_quote": "<verbatim, copied from one candidate>",
  "confidence": "high",
  "rationale": "<one sentence>"
}}
""".strip()
