# LLM_clinical_annotations

This repo contains clinical-note annotation pipelines that use LLMs and related
note-preparation utilities.

## Layout

- `shared/` - common note cleaning, note loading, snippet packing, Azure OpenAI
  calls, JSON parsing, and shared preprocessing scripts.
- `cancer_stage/` - cancer stage note extraction.
- `binary_NEPC/` - patient-level NEPC / AVPC / biomarker / conventional
  classifier.
- `longitudinal_NEPC/` - AVPC / NEPC criteria onset timeline extraction.
- `gleason_score/` - Gleason / Grade Group timeline extraction.

## Common Data Source

Most downstream applications read the shared `prostate_text_data.csv` note
source. Build it with:

```bash
python shared/compile_prostate_notes.py
```

By default, that command reads raw OncDRS notes and derives the cohort from
raw OncDRS ICDs using the COMPASS ICD-based definition:
patients with ICD-10 `C61`, excluding patients with a competing non-prostate
primary ICD. By default it reads the raw OncDRS diagnosis source
`/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/EHR_DIAGNOSIS.csv`.

The default data root is `/data/gusev/USERS/jpconnor/data/LLM_annotations/`.
Override it with `LLM_ANNOTATIONS_DATA_PATH`; the legacy `CAIA_COMPASS_DATA_PATH`
is still accepted as a fallback.

## LLM Configuration

LLM applications use Azure OpenAI AAD authentication via
`DefaultAzureCredential`. Common overrides:

```text
LLM_ANNOTATIONS_DATA_PATH
BINARY_NEPC_OUTPUT_DIR
CAIA_AZURE_OPENAI_ENDPOINT
CAIA_AZURE_OPENAI_API_VERSION
CAIA_AZURE_OPENAI_MODEL
```

The old `CAIA_COMPASS_*` data/output environment variables are still honored for
compatibility.
