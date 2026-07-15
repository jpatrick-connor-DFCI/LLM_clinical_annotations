# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the **Vertex AI variant** of the LLM clinical annotations pipeline. It is
structurally identical to the parent `LLM_clinical_annotations/` repo except that the
LLM client layer in `shared/llm_helpers.py` calls Google Vertex AI (Gemini) instead of
the DFCI Azure OpenAI endpoint.

## Setup

```bash
pip install -r requirements.txt
gcloud auth application-default login
```

No API key is needed. Auth is handled by Application Default Credentials (ADC).

## Commands

```bash
# Build the shared note source (required before most LLM pipelines)
python shared/compile_prostate_notes.py --derive-prostate-mrns

# Binary NEPC classifier
python binary_NEPC/run_NEPC_classifier.py --mrn-file /path/to/mrns.txt --output-dir /path/to/out

# Cancer stage extraction (two-step)
python cancer_stage/extract_stage_notes.py --output-dir /path/to/out   # Step 1: scan
python cancer_stage/run_stage_extraction.py --output-dir /path/to/out  # Step 2: LLM

# Pilot / subset run
python cancer_stage/run_stage_extraction.py --mrns "12345,67890" --limit-patients 10
```

All scripts are run from this `vertex_ai/` directory. They patch `sys.path` themselves,
so no install step is needed beyond `pip install -r requirements.txt`.

## LLM Configuration

| Env var | Default | Notes |
|---|---|---|
| `VERTEX_PROJECT` | `gusevlabllm` | GCP project |
| `VERTEX_LOCATION` | `us-central1` | GCP region |
| `VERTEX_MODEL` | `gemini-2.0-flash-001` | Override to use a different Gemini model |
| `LLM_ANNOTATIONS_DATA_PATH` | `/data/gusev/USERS/jpconnor/data/LLM_annotations/` | Note data root |
| `BINARY_NEPC_OUTPUT_DIR` | `<data_path>/LLM_NEPC_labels/` | NEPC classifier output |
| `STAGE_OUTPUT_DIR` | `/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/` | Stage pipeline output |

## Architecture

### What changed vs. the Azure version

Only `shared/llm_helpers.py` differs from the parent repo:

- **`build_client()`** calls `vertexai.init(project, location)` and returns `None`
  (Gemini models are stateless; `GenerativeModel` is instantiated per call).
- **`call_with_retry()`** maps the OpenAI-style `messages` list to Gemini's
  `system_instruction` + `generate_content()` pattern. Uses
  `response_mime_type="application/json"` for constrained JSON output.
  Error handling maps GCP exceptions (`ResourceExhausted`, `DeadlineExceeded`,
  `GoogleAPIError`) to the same retry logic the Azure version used for rate limits,
  timeouts, and API errors.
- Imports: `google-cloud-aiplatform` replaces `openai` + `azure-identity`.

Everything else — note loading, trigger regexes, snippet building, prompts, pipeline
orchestration, output formats — is identical to the Azure version.

### Pipeline pattern

Every task module follows the same two-step structure:

1. **Scan step** (`extract_*.py`) — regex trigger matching across notes, context-window
   extraction, write evidence TSV. Resumable: skips already-scanned files.
2. **LLM step** (`run_*.py`) — reads evidence TSV, packs snippets into per-patient
   payload-sized chunks, calls Gemini once per chunk via `ThreadPoolExecutor`, writes
   raw findings + processed log incrementally (resumable with `--overwrite` opt-in).

### `shared/` modules

- **`llm_helpers.py`** — Vertex AI client, note loading, snippet building, NEPC prompt
  and trigger regexes. All other modules import from here.
- **`longitudinal_helpers.py`** — per-note dedup, patient chunking, and date resolution
  for the timeline pipelines (Gleason, AVPC/NEPC, cancer stage). Inherits
  `build_client` and `call_with_retry` from `llm_helpers`.
- **`utils.py`** — `clean_note(text, note_type)`: universal + NOTE_TYPE-specific regex
  cleaning (Clinician / Imaging / Pathology).
- **`compile_prostate_notes.py`** — builds `prostate_text_data.csv`, the default note
  source for all LLM pipelines.
