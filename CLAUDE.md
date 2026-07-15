# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (no test suite exists)
pip install -r requirements.txt

# Build the shared note source (required before most pipelines)
python shared/compile_prostate_notes.py --derive-prostate-mrns

# Binary NEPC classifier
python binary_NEPC/run_NEPC_classifier.py --mrn-file /path/to/mrns.txt --output-dir /path/to/out

# Cancer stage extraction (two-step)
python cancer_stage/extract_stage_notes.py --output-dir /path/to/out   # Step 1: scan
python cancer_stage/run_stage_extraction.py --output-dir /path/to/out  # Step 2: LLM

# Pilot / subset run (most scripts support these flags)
python cancer_stage/run_stage_extraction.py --mrns "12345,67890" --limit-patients 10
```

All scripts are run from the repo root. They patch `sys.path` themselves, so no install step is needed beyond `pip install -r requirements.txt`.

## Architecture

### Data flow

```
OncDRS raw JSONs  →  compile_prostate_notes.py  →  prostate_text_data.csv
                                                          │
                                        load_notes() (llm_helpers.py)
                                                          │
                                       clean_note()  ←  shared/utils.py
                                                          │
                                    trigger regex scan → snippet extraction
                                                          │
                                      patient chunking → LLM calls (Azure OpenAI)
                                                          │
                                   incremental TSV writes → final dedup/timeline
```

### `shared/` — the backbone

- **`llm_helpers.py`** — everything the binary NEPC classifier needs: Azure OpenAI client (`build_client`/`call_with_retry`), note loading (`load_notes` with three-tiered precedence: bundle → CSV → raw JSONs), snippet building, and the NEPC system prompt and trigger regexes. Also owns path/env-var constants.
- **`longitudinal_helpers.py`** — extends `llm_helpers` for the timeline pipelines (Gleason, AVPC/NEPC timeline, cancer stage). Adds `iter_note_snippets` (per-note dedup, preserves earliest date), `group_patient_snippets` (greedy packing into payload-sized chunks), and `resolve_date`.
- **`utils.py`** — `clean_note(text, note_type)`: applies universal regex rules then NOTE_TYPE-specific rules (Clinician / Imaging / Pathology).
- **`compile_prostate_notes.py`** — builds `prostate_text_data.csv`, the default note source for all pipelines.

### Pipeline pattern

Every task module follows the same structure:

1. **Scan step** (`extract_*.py`) — regex trigger matching across notes, context-window extraction, write evidence TSV. Uses `ProcessPoolExecutor` over raw files for parallelism. Resumable: skips already-scanned files.
2. **LLM step** (`run_*.py`) — reads evidence TSV, groups snippets into per-patient chunks (greedy, up to `payload_max_chars`), calls LLM once per chunk via `ThreadPoolExecutor`, writes raw findings + processed log incrementally (so runs are resumable with `--overwrite` opt-in).

Patient chunking is lossless: patients with many notes get multiple LLM calls rather than truncation, ensuring rare findings are never silently dropped.

### Key constants (all overridable via env vars)

| Env var | Default |
|---|---|
| `LLM_ANNOTATIONS_DATA_PATH` | `/data/gusev/USERS/jpconnor/data/LLM_annotations/` |
| `BINARY_NEPC_OUTPUT_DIR` | `<data_path>/LLM_NEPC_labels/` |
| `STAGE_OUTPUT_DIR` | `/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/` |
| `CAIA_AZURE_OPENAI_ENDPOINT` | DFCI Azure OpenAI endpoint |
| `CAIA_AZURE_OPENAI_API_VERSION` | `2024-04-01-preview` |
| `CAIA_AZURE_OPENAI_MODEL` | `gpt-4o` |

The legacy `CAIA_COMPASS_*` env vars are still accepted as fallbacks.

### LLM authentication

Uses `DefaultAzureCredential` (AAD token). No API key is needed; the credential resolves automatically in the DFCI environment (e.g., Azure CLI login or managed identity).

### Note types

Notes are classified as `Clinician`, `Imaging`, or `Pathology` based on filename patterns in the OncDRS source files. This type label drives both cleaning rules (`shared/utils.py`) and snippet-selection heuristics.
