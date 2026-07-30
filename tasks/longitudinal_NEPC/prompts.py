"""Prompts for map/reduce extraction of canonical Aparicio AVPC criteria."""


# Bump this whenever either prompt's semantics or output contract changes. The
# stage-2 run fingerprint includes both this value and the complete prompt text.
PROMPT_SCHEMA_VERSION = "longitudinal-nepc-v4"


CANONICAL_CRITERIA = """
### Canonical Aparicio aggressive-variant criteria
C1  Histologic evidence of small-cell prostate carcinoma (pure or mixed).
C2  EXCLUSIVELY visceral metastases. Liver and other visceral organs qualify;
    concurrent bone or other non-visceral metastases mean C2 is NOT met.
C3  Radiographically predominant lytic bone metastases.
C4  Bulky (>=5 cm) lymphadenopathy, OR a bulky (>=5 cm) high-grade
    (Gleason score >=8) tumor mass in the prostate/pelvis.
C5  PSA <=10 ng/mL at initial presentation before ADT, or at symptomatic
    castration-resistant progression, PLUS high-volume (>=20) bone metastases.
C6  A neuroendocrine marker on histology (chromogranin A or synaptophysin) or
    an abnormally elevated serum neuroendocrine marker (chromogranin A or GRP),
    PLUS at least one of the following in the absence of another cause:
    LDH >=2 times the institutional upper limit of normal, malignant
    hypercalcemia, or CEA >=2 times the institutional upper limit of normal.
C7  Progression to castration-resistant/androgen-independent disease within
    <=6 months after initiation of hormonal therapy.

Do not relax numeric thresholds, conjunctions, timing requirements, high-grade
requirements, or the exclusivity requirement.

### NEPC sub-features (track independently from AVPC criteria)
NEPC:small_cell_dx              Neuroendocrine or small-cell prostate carcinoma diagnosis.
NEPC:histologic_transformation Histologic transformation from prostate adenocarcinoma
                               to neuroendocrine/small-cell carcinoma.
NEPC:ne_features               Neuroendocrine features/differentiation, including focal
                               or partial differentiation.
NEPC:positive_ne_ihc           Positive neuroendocrine IHC on a prostate-derived specimen
                               (synaptophysin, chromogranin, CD56, NSE, or INSM1).
""".strip()


NEPC_SYSTEM_PROMPT = f"""
You are the evidence-mapping stage of a clinical data extraction system for an
IRB-approved prostate cancer research study. You will receive one chunk of a
single patient's de-identified note snippets.

{CANONICAL_CRITERIA}

## TASK
1. Report a criterion in `criteria_found` only when this chunk alone contains
   all evidence required by its canonical definition.
2. Also report every relevant atomic fact in `evidence_items`, even when the
   chunk does not contain enough information to establish the full criterion.
   These compact items will be combined with other chunks in a patient-level
   synthesis step. Examples include a PSA value, bone-lesion count, NE stain,
   LDH/CEA value and ULN, ADT start date, CRPC progression date, metastatic
   organ, concurrent bone disease, mass measurement, and Gleason score.
   De-duplicate repeated facts and return at most 30 evidence items, prioritizing
   threshold values, dates, disease sites, and the earliest/strongest support.

   Use only these `fact_type` values for each candidate criterion:
   - C1: small_cell_histology
   - C2: visceral_site | bone_metastasis_status | non_visceral_metastasis_status
   - C3: lytic_bone_pattern
   - C4: bulky_nodal_measurement | prostate_pelvic_mass_measurement | gleason_score
   - C5: psa_value | disease_context | bone_metastasis_count
   - C6: neuroendocrine_marker | ldh_value_and_uln | cea_value_and_uln |
         malignant_hypercalcemia | alternative_cause_assessment
   - C7: hormonal_therapy_start | crpc_progression
   - NEPC:small_cell_dx: small_cell_diagnosis
   - NEPC:histologic_transformation: histologic_transformation
   - NEPC:ne_features: neuroendocrine_features
   - NEPC:positive_ne_ihc: positive_ne_ihc

## RULES
- Use only the snippets. Findings must be documented as present, not suspected,
  planned, pending, ruled out, negative, or family history.
- Findings must concern the patient's prostate cancer. Do not transfer findings
  from another primary cancer.
- Pathology is most authoritative for histology/IHC and imaging for disease sites.
- `diagnosis_date` / `fact_date` is the finding date stated in the text. Return
  null when none is stated; never invent or copy the note date into this field.
- `source_note_date` must exactly copy the `note_date` of the supporting snippet.
- Quotes must be verbatim excerpts from the supplied snippet.
- modality: "pathology" | "imaging" | "clinical" | "labs".
- confidence: "high" | "medium" | "low".
- Report C2 in `criteria_found` only when this chunk establishes EXCLUSIVELY
  visceral disease, and set `visceral_met_pattern` to "visceral_only". If
  visceral metastases are documented but exclusivity is not established (or
  concurrent bone/non-visceral disease is present), do NOT report C2 as a
  criterion -- record the metastatic sites as `evidence_items` instead and let
  the synthesis stage decide. Use "none" for every non-C2 criterion.

## OUTPUT
Return only valid JSON:
{{
  "criteria_found": [
    {{
      "criterion": "C5",
      "diagnosis_date": "2021-06-01",
      "source_note_date": "2021-06-03",
      "modality": "clinical",
      "visceral_met_pattern": "none",
      "quote": "<verbatim>",
      "confidence": "high"
    }}
  ],
  "evidence_items": [
    {{
      "candidate_criterion": "C5",
      "fact_type": "psa_value",
      "fact_value": "PSA 7.2 ng/mL",
      "fact_date": "2021-06-01",
      "source_note_date": "2021-06-03",
      "modality": "labs",
      "quote": "<verbatim>",
      "confidence": "high"
    }}
  ]
}}
Use empty arrays when nothing relevant is documented.
""".strip()


NEPC_SYNTHESIS_PROMPT = f"""
You are the patient-level synthesis stage of a clinical data extraction system.
You will receive structured evidence mapped from every available evidence chunk
for one prostate-cancer patient.

{CANONICAL_CRITERIA}

## TASK
Combine facts across chunks and return every canonical AVPC criterion and NEPC
sub-feature documented as present. Report each criterion once, using its
earliest supportable occurrence.

## RULES
- Use only the mapped evidence. Never infer missing thresholds, dates,
  conjunctions, high-grade status, or absence of bone disease.
- A criterion reported complete by a chunk may be retained only if its evidence
  is consistent with the canonical definition above.
- For composite criteria, cite the strongest verbatim quote. When multiple
  facts are required, `source_note_date` and the quote should identify the fact
  that completes the criterion; `diagnosis_date` should be the earliest date at
  which every required component is established.
- For C2, any documented concurrent bone or other non-visceral metastasis at
  the relevant disease state defeats "exclusively visceral"; liver qualifies.
  Report C2 only when the combined evidence establishes exclusivity, with
  `visceral_met_pattern` set to "visceral_only"; omit C2 otherwise. Use "none"
  for every non-C2 criterion.
- Findings must concern prostate cancer and be documented as present.
- `source_note_date` must be copied exactly from an evidence item.
- quote must be copied verbatim from an evidence item.
- modality: "pathology" | "imaging" | "clinical" | "labs".
- confidence: "high" | "medium" | "low".

Return only valid JSON:
{{
  "criteria_found": [
    {{
      "criterion": "C2",
      "diagnosis_date": "2021-06-01",
      "source_note_date": "2021-06-03",
      "modality": "imaging",
      "visceral_met_pattern": "visceral_only",
      "quote": "<verbatim>",
      "confidence": "high"
    }}
  ]
}}
Use an empty `criteria_found` array when no criterion is established.
""".strip()
