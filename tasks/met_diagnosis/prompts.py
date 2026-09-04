"""Prompts for the metastatic prostate cancer label + first-mention date task.

Two stages: a per-chunk map that proposes candidate metastatic-disease
statements, and a per-patient adjudication that returns one final yes/no plus
the earliest qualifying evidence.

Bump PROMPT_SCHEMA_VERSION on any semantic or output-contract change; the
stage-2 run fingerprint hashes this value and both prompt texts, so a changed
prompt forces --overwrite rather than mixing output generations.
"""

PROMPT_SCHEMA_VERSION = "met-dx-v1"


QUALIFYING_DEFINITION = """
### What qualifies as metastatic prostate cancer
A qualifying statement asserts that THIS patient's prostate cancer has spread to
a DISTANT site. Any of these qualify:
  (a) An asserted statement of metastatic prostate cancer in general terms:
      "metastatic prostate cancer", "mHSPC", "mCRPC", "metastatic castration-
      resistant prostate cancer", "de novo metastatic disease", "widespread
      metastatic disease".
  (b) An asserted distant metastasis at a named site:
      - BONE / osseous / skeletal: "osseous metastases", "bone mets",
        "metastatic disease to the spine", "sclerotic lesions consistent with
        osseous metastases".
      - VISCERAL: liver/hepatic, lung/pulmonary, adrenal, brain, pleural,
        peritoneal.
      - DISTANT (non-regional) NODAL: retroperitoneal, para-aortic,
        mediastinal, supraclavicular, cervical, inguinal nodes.
  (c) An asserted M category of M1, M1a, M1b, or M1c, or an asserted
      stage IV / stage 4 prostate cancer.
  (d) A pathology report of a metastatic deposit of prostatic origin (for
      example a bone or node biopsy read as metastatic adenocarcinoma
      consistent with prostate primary).

### What does NOT qualify (report nothing for these)
- Negated statements: "no evidence of metastatic disease", "no osseous
  metastases", "bone scan negative for metastatic disease", "no distant
  spread", "without evidence of metastasis".
- Hedged, suspected, or equivocal wording: "concerning for metastasis",
  "suspicious for osseous metastatic disease", "cannot exclude metastasis",
  "equivocal", "indeterminate", "possible", "probable", "favor degenerative
  change", "versus metastasis", "differential includes metastasis".
- REGIONAL PELVIC NODAL DISEASE ALONE. This is the sharpest boundary in this
  task. Prostate cancer involving only pelvic, obturator, internal/external
  iliac, perirectal, or periprostatic nodes is N1 disease, NOT M1, and does NOT
  qualify. "Metastatic adenocarcinoma in 2 of 14 pelvic lymph nodes" from a
  prostatectomy specimen is N1 and does NOT qualify, despite the word
  "metastatic". Only report nodal disease when the involved nodes are named as
  distant/non-regional (retroperitoneal, para-aortic, mediastinal,
  supraclavicular), or when the statement independently asserts metastatic
  disease elsewhere.
- Benign or degenerative imaging findings: "degenerative changes", "sclerotic
  focus likely a bone island", "arthritic changes", "age-related changes",
  "postsurgical change", "healing fracture".
- Metastases attributed to a DIFFERENT primary cancer (lung, bladder,
  colorectal, renal, melanoma), or metastatic disease in the FAMILY history.
- Surveillance, risk, or future framing: "monitoring for metastatic
  progression", "at risk for metastatic disease", "if he develops metastases",
  "restaging to rule out metastasis", "pending bone scan".
- Stock/boilerplate text describing a study POPULATION rather than this
  patient: clinical trial eligibility criteria ("inclusion criteria: patients
  with metastatic castration-resistant prostate cancer"), protocol titles and
  study descriptions ("a Phase III study of ... in metastatic prostate
  cancer"), consent forms, cohort definitions, and registry or questionnaire
  templates. Such text names metastatic disease because the trial targets it,
  not because the patient has it. A patient being screened for, consented to,
  or enrolled on such a trial is NOT evidence of metastatic disease -- report
  the underlying clinical or imaging assertion if one is separately stated, and
  otherwise report nothing.

### Where a qualifying assertion may appear
Imaging reports, pathology reports, and clinician progress notes are ALL
acceptable sources, and none outranks the others. Record which one it was in
`modality`.

- `modality: "imaging"` -- a radiology IMPRESSION or findings section asserting
  metastatic disease: "IMPRESSION: multiple osseous metastases", "innumerable
  sclerotic osseous metastases", "new hepatic metastasis". A DEFINITIVE imaging
  assertion qualifies. A hedged one ("suspicious for", "concerning for",
  "cannot exclude") does NOT, no matter how strongly worded the rest of the
  report is. This distinction matters more here than anywhere else in the task:
  radiologists hedge constantly, and hedged language is not a diagnosis.
- `modality: "pathology"` -- a biopsy or resection of a metastatic deposit.
- `modality: "clinical"` -- an oncologist's assessment, impression, problem
  list, or oncologic history asserting the disease as established: "ASSESSMENT:
  metastatic castration-resistant prostate cancer", "Problem list: metastatic
  prostate cancer to bone", "s/p radium-223 for osseous metastases", "he has
  widely metastatic disease".

The same exclusions apply regardless of source: a clinician's hedged or
surveillance wording does NOT qualify, exactly as it would not in a radiology
report.

When in doubt, report nothing.
""".strip()


MET_DX_MAP_PROMPT = f"""
You are the candidate-identification stage of a clinical data extraction system
for an IRB-approved prostate cancer research study. You will receive one chunk
of a single patient's de-identified note snippets.

{QUALIFYING_DEFINITION}

## TASK
Return every statement in this chunk that is a candidate qualifying assertion of
metastatic prostate cancer, with a verbatim quote. Report each distinct
statement once; do not emit one entry per repetition of the same copy-forward
sentence.

A later adjudication stage reviews all candidates for this patient, so report a
candidate whenever a statement plausibly qualifies, and record honestly in
`assertion_type` and `confidence` how firmly it is asserted. Do NOT report
anything matching the "does NOT qualify" list above.

## RULES
- Use only the supplied snippets. Never infer metastatic disease from treatment
  choice (for example ADT, radium-223, or docetaxel), from a rising PSA, or from
  the fact that a scan was ordered.
- The disease must be the patient's own prostate cancer.
- `quote` must be copied verbatim from one supplied snippet, and must itself
  contain both the metastasis term and the words that assert it. Do not stitch
  together text from two places. Keep the quote to one or two sentences.
- `source_note_date` must exactly copy the `note_date` of the snippet the quote
  came from.
- `stated_metastasis_date` is a date explicitly stated in the text for when the
  metastatic disease was found or diagnosed (for example "bone metastases first
  identified in March 2021"). Return null when no date is stated in the text;
  NEVER copy the note date into this field. Dates must be valid `YYYY-MM-DD`:
  normalize a stated year to January 1 and a stated year-month to the first of
  that month. Never emit `xx`, `00`, seasons, ranges, or impossible calendar
  dates.
- `evidence_type`: "stated_metastatic_disease" (a general assertion, as in (a))
  | "imaging_metastasis" (a radiologic assertion at a site, as in (b)) |
  "pathologic_metastasis" (a tissue diagnosis, as in (d)) | "m_stage" (an
  asserted M1* or stage IV, as in (c)).
- `met_site`: "bone" | "visceral" | "distant_nodal" | "unspecified". Use
  "unspecified" only when the statement asserts metastatic disease without
  naming a site. The site must be named in the quote itself for anything other
  than "unspecified".
- `assertion_type`: "established" (asserted as the patient's disease) |
  "reported_history" (attributed to an outside record or a prior note).
- `modality`: "imaging" | "pathology" | "clinical".
- `confidence`: "high" | "medium" | "low".

## OUTPUT
Return only valid JSON:
{{
  "candidates": [
    {{
      "evidence_type": "imaging_metastasis",
      "assertion_type": "established",
      "met_site": "bone",
      "stated_metastasis_date": null,
      "source_note_date": "2021-04-12",
      "modality": "imaging",
      "quote": "<verbatim>",
      "confidence": "high"
    }}
  ]
}}
Return {{"candidates": []}} when the chunk contains no qualifying statement.
""".strip()


MET_DX_SYNTHESIS_PROMPT = f"""
You are the adjudication stage of a clinical data extraction system. You will
receive every candidate metastatic-disease statement extracted from one prostate
cancer patient's notes, across all evidence chunks.

{QUALIFYING_DEFINITION}

## TASK
Decide ONCE for this patient whether the record establishes qualifying
metastatic prostate cancer, and if so, identify the EARLIEST qualifying
statement.

Adjudicate against the definition above, not against the candidate count. Many
weak candidates do not add up to one qualifying assertion; a single unambiguous
statement is sufficient, whether it comes from imaging, pathology, or a
clinician's note. Reject the patient when every candidate is negated, hedged,
benign/degenerative, regional pelvic nodal disease only, attributable to a
non-prostate primary, or framed as surveillance or risk.

Do not prefer one modality over another when both are unambiguous -- select
whichever is EARLIEST. A patient whose only qualifying evidence is a
radiologist's definitive impression is a positive, as is one whose only
qualifying evidence is a clinician's stated assessment.

## RULES
- Use only the supplied candidates. Never introduce a quote, a date, or a fact
  that is not present in them.
- `supporting_quote` and `source_note_date` must be copied EXACTLY, and as a
  matched pair, from the single candidate you select.
- Select the candidate that is the earliest qualifying evidence of metastatic
  disease, ranking by stated metastasis date when one exists and otherwise by
  source note date.
- `metastasis_date` is the date stated in that candidate's text. Return null
  when the selected candidate states no date -- a downstream step falls back to
  the earliest qualifying note date and records that it did so. Do not guess.
- When `has_metastatic_disease` is false, set every other field to null and give
  a one-sentence `rationale` naming the disqualifying reason.

## OUTPUT
Return only valid JSON:
{{
  "has_metastatic_disease": true,
  "evidence_type": "imaging_metastasis",
  "assertion_type": "established",
  "met_site": "bone",
  "metastasis_date": null,
  "source_note_date": "2021-04-12",
  "modality": "imaging",
  "supporting_quote": "<verbatim, copied from one candidate>",
  "confidence": "high",
  "rationale": "<one sentence>"
}}
""".strip()
