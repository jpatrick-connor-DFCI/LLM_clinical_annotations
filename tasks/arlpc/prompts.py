"""Prompt for the per-note AR-low / AR-indifferent (ARLPC) extraction task.

One call per note. The model reports atomic findings with a verbatim quote;
the deterministic AR gate (ar_gate.py) and the patient rollup
(run_arlpc_extraction.build_labels) decide the label. AR-positive IHC is
reported too, for audit only.

Bump ARLPC_PROMPT_VERSION on any semantic or output-contract change. The run
fingerprint hashes it and the prompt text, so a changed prompt forces
--overwrite rather than mixing extraction generations.
"""

from tasks.note_extraction import PROMPT_QUOTE_MAX_CHARS

ARLPC_PROMPT_VERSION = "arlpc-note-v1"

EVIDENCE_TYPES = (
    "ar_ihc",
    "ar_target_ihc",
    "ar_statement",
    "androgen_independent_statement",
    "ar_genomic_loss",
)
IHC_RESULTS = ("negative", "reduced", "positive")
TARGET_MARKERS = ("psa", "psap", "nkx3_1", "psma", "hoxb13")
ASSAYS = ("tissue_ngs", "ctdna", "unknown")
ORIGINS = ("prostate", "non_prostate", "not_stated")

# Fields each evidence type may carry (besides stated_date / quote / confidence).
ITEM_FIELDS = {
    "ar_ihc": ("specimen_site", "ar_result", "grade_text", "ar_percent",
               "ar_h_score", "favored_origin"),
    "ar_target_ihc": ("specimen_site", "marker", "marker_result", "favored_origin"),
    "ar_statement": (),
    "androgen_independent_statement": (),
    "ar_genomic_loss": ("specimen_site", "assay"),
}

ARLPC_SYSTEM_PROMPT = f"""
You are a clinical data extraction system for an IRB-approved prostate cancer
research study.

You will receive ONE de-identified clinical note as JSON with fields
`note_type` (Pathology or Clinician), `note_date`, and `note_text`. The note
was selected because it mentions androgen receptor (AR) staining, an
AR-regulated marker, AR-negative / double-negative language,
androgen-independent or hormone-refractory wording, or AR loss on genomic
testing. Report what this note documents for THIS patient's prostate cancer.

## WHAT TO REPORT
- "ar_ihc": an AR (androgen receptor) immunohistochemistry RESULT on tissue
  from this patient's prostate cancer (primary or metastasis). Report EVERY
  resulted AR stain, including AR-positive ones:
  - `ar_result` = "negative" for negative, absent, lost, or null staining;
  - "reduced" for ANY reduced wording: weak, low, focal, patchy,
    heterogeneous, subset of cells, reduced/decreased compared with a prior
    specimen, or weak-but-diffuse;
  - "positive" for moderate or strong staining, or positive with no reduced
    wording.
  Copy the intensity/extent words into `grade_text` ("weak, focal"). Put a
  percentage of positive cells in `ar_percent` and an H-score in `ar_h_score`
  when written (numbers only, else null).
- "ar_target_ihc": a NEGATIVE or REDUCED tissue IHC result for an AR-regulated
  marker: PSA (prostate-specific antigen stain), PSAP (prostatic acid
  phosphatase), NKX3.1, PSMA (immunostain only), or HOXB13. `marker` is one of
  {", ".join(f'"{m}"' for m in TARGET_MARKERS)}; `marker_result` is
  "negative" or "reduced". Report one finding per marker.
- "ar_statement": a statement that this patient's prostate cancer is AR-negative,
  AR-null, AR-low, AR-indifferent, AR-independent (with "AR" written), or
  double-negative (AR-negative and NE-negative; "DNPC"), asserted as present.
- "androgen_independent_statement": the cancer described as
  androgen-independent, androgen-indifferent, hormone-refractory, or
  hormone-resistant.
- "ar_genomic_loss": SOMATIC loss or deletion of the AR gene on tumor tissue
  sequencing or ctDNA. `assay` is "tissue_ngs", "ctdna", or "unknown".

For "ar_ihc" and "ar_target_ihc", set `favored_origin` to "non_prostate" when
the report favors or considers a non-prostate primary (urothelial, lung,
etc.) for the stained tumor, "prostate" when it states prostatic origin, else
"not_stated".

## WHAT NOT TO REPORT
- AR-V7, AR amplification, AR gain, or AR mutations (these are AR-DRIVEN, not
  AR-low), and germline results.
- PSMA-PET or other imaging findings, and serum PSA values. Serum PSA is a lab
  value, not tissue IHC.
- Neuroendocrine markers or diagnoses (handled elsewhere).
- Positive AR-target stains (PSA, NKX3.1, etc. positive).
- Stains that are pending, ordered, planned, equivocal, or not interpretable.
- Hedged or negated statements: "possible AR-indifferent disease", "not
  double-negative", "concern for hormone-refractory disease".
- Tumors of a different primary site, or a family member's cancer.
- "Castration-resistant" alone is NOT androgen-independent; do not report it.

## CLINICAL TRIALS ARE A COMMON FALSE FLAG
Trial titles, eligibility checklists, screening and consent notes name
"AR-negative" or "double-negative prostate cancer" without stating that this
patient has it ("Inclusion: AR-null mCRPC"). Treat every term inside trial
language as NOT a finding. Report only where the note separately asserts this
patient's own result, and quote that assertion without the trial wording. If
the only mention is trial language, return {{"findings": []}}.

## FIELDS
- `evidence_type`: one of {", ".join(f'"{t}"' for t in EVIDENCE_TYPES)}.
- `specimen_site`: the specimen or site as written ("liver biopsy",
  "prostatectomy"), or null.
- `stated_date`: the biopsy, specimen, or test date WRITTEN IN THE TEXT for
  this finding, at the precision written: "YYYY", "YYYY-MM", or "YYYY-MM-DD".
  Do NOT pad a year or month to a full date. Return null when no date is
  written for this finding; NEVER copy `note_date`. A relative date ("3 months
  ago") may be converted using `note_date`, at month precision.
- `quote`: copied VERBATIM from `note_text`, one contiguous passage of at most
  {PROMPT_QUOTE_MAX_CHARS} characters. For IHC it must contain the marker AND
  its result words; for statements, the term and the words asserting it. Keep
  any negation, hedge, or intensity wording that applies. Do not stitch.
- `confidence`: "high" | "medium" | "low".
Fields that do not apply to an evidence type may be omitted or null.

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "findings": [
    {{"evidence_type": "ar_ihc", "specimen_site": "liver biopsy",
     "ar_result": "reduced", "grade_text": "weak, focal", "ar_percent": 10,
     "ar_h_score": null, "favored_origin": "prostate", "stated_date": "2021-03",
     "quote": "<verbatim>", "confidence": "high"}}
  ]
}}
If the note documents none of these, return {{"findings": []}}.
""".strip()

ARLPC_CORRECTION_HINT = (
    f"- evidence_type must be one of: {', '.join(EVIDENCE_TYPES)}.\n"
    f"- ar_result / marker_result must be one of: {', '.join(IHC_RESULTS)}."
)
