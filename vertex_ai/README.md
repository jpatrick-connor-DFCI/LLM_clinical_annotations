# Vertex AI clinical annotation pipelines

This wrapper contains the Google Vertex AI / Gemini clinical-note annotation
pipelines. Run commands from this `vertex_ai/` directory.

## Setup

```bash
python -m pip install -r requirements.txt
gcloud auth application-default login
```

The default GCP project is `gusevlabllm`. Provider settings can be overridden
with:

```text
VERTEX_PROJECT
VERTEX_LOCATION
VERTEX_MODEL
```

## Layout

- `shared/` — note processing and the Vertex AI client layer.
- `binary_NEPC/` — patient-level NEPC / AVPC / biomarker classifier.
- `cancer_stage/` — cancer-stage note extraction.
- `longitudinal_NEPC/` — AVPC / NEPC timeline extraction.
- `gleason_score/` — Gleason / Grade Group timeline extraction.

For the binary NEPC notebook, open
`binary_NEPC/generate_notes_and_run_llm.ipynb`, review the data paths, and
enable the desired run toggles.
