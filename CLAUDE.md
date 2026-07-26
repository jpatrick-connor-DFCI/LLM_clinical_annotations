# CLAUDE.md

Guidance for Claude Code when working in this repository. All commands are run
from the repo root.

## Commands

```bash
# Install (core + one provider's SDK)
pip install -e ".[dfci_gpt]"     # or ".[vertex_ai]"

# Build the shared note source (required before most pipelines)
python preprocessing/cli/compile_prostate_notes.py --derive-prostate-mrns

# Binary NEPC: compile snippets, then classify
python preprocessing/cli/compile_patient_snippets.py --output-path /path/to/patient_snippets.json.gz
python tasks/binary_NEPC/run_NEPC_classifier.py --snippets-path /path/to/patient_snippets.json.gz --output-dir /path/to/out --provider dfci_gpt

# Cancer stage / Gleason / longitudinal NEPC: collect evidence, then extract
python preprocessing/cli/extract_stage_notes.py --output-dir /path/to/out          # stage: scan (no LLM)
python tasks/cancer_stage/run_stage_extraction.py --output-dir /path/to/out --provider vertex_ai

python preprocessing/cli/collect_gleason_notes.py --output-dir /path/to/out
python tasks/gleason_score/build_gleason_timeline.py --output-dir /path/to/out --provider dfci_gpt

python preprocessing/cli/collect_nepc_notes.py --output-dir /path/to/out
python tasks/longitudinal_NEPC/build_nepc_timeline.py --output-dir /path/to/out --provider dfci_gpt

# Pilot / subset run (most task runners support these)
python tasks/cancer_stage/run_stage_extraction.py --mrns "12345,67890" --provider dfci_gpt
```

Scripts patch `sys.path` themselves (each inserts the repo root before its
`preprocessing`/`providers`/`tasks` imports), so they run directly without
`pip install -e .` — that's only needed to import those packages from a
notebook or the REPL.

## Architecture

### Provider-agnostic core, thin provider adapters

`preprocessing/` never imports a provider SDK (`openai`, `azure`, `vertexai`,
`google.genai`). All LLM-calling logic lives behind `providers.get_provider(name)`,
which lazily imports only the requested adapter's SDK. Task runners take
`--provider {dfci_gpt,vertex_ai}` and `--model` (defaults to the provider's
`default_model`).

Adding a third provider means writing one new `providers/<name>.py` exposing
`name`, `default_model`, `build_client()`, `call_with_retry(client, model_name,
messages, max_retries=3) -> (text, error)`, and registering it in
`providers/__init__.py`. Nothing in `preprocessing/` or `tasks/` changes.

### Data flow

```
OncDRS raw JSONs → preprocessing/cli/compile_prostate_notes.py → prostate_text_data.csv
                                          │
                          preprocessing.notes.load_notes()
             (precedence: explicit bundle > compiled CSV > raw OncDRS JSONs)
                                          │
                       preprocessing.utils.clean_note()
                                          │
                preprocessing.triggers: trigger regex scan → snippet extraction
                                          │
              preprocessing.snippets / preprocessing.longitudinal: patient chunking
                                          │
                    tasks/<task>/*.py: LLM calls via providers.get_provider(...)
                                          │
                    incremental TSV writes → final dedup / timeline build
```

### Two-phase task pattern

Every task is split into a **preprocessing** step and a **task runner**:

1. **Preprocessing** (`preprocessing/cli/`) — provider-independent. Regex
   trigger matching across notes, context-window extraction, writes an
   evidence/snippet artifact (TSV or `.json.gz` bundle). `compile_patient_snippets.py`
   and `collect_gleason_notes.py`/`collect_nepc_notes.py` use
   `ProcessPoolExecutor` for the per-note scan; `extract_stage_notes.py`
   additionally parallelizes over raw files and resumes (skips
   already-scanned files) when re-run without `--overwrite`.
2. **Task runner** (`tasks/<task>/`) — reads the evidence artifact, groups
   snippets into per-patient chunks (greedy packing up to `payload_max_chars`),
   calls the selected provider once per chunk via `ThreadPoolExecutor`, writes
   raw findings + a processed-patient log incrementally (resumable; re-run
   without `--overwrite` skips already-processed patients). `run_NEPC_classifier.py`
   additionally supports `--retry-failures` to retry only prior failures.

Patient chunking is lossless: patients with many notes get multiple LLM calls
rather than truncation, so rare findings are never silently dropped.

### Snippet sizing

`preprocessing/config.py` defines `SnippetProfile(context_chars, max_chars,
payload_max_chars)` and `SNIPPET_PROFILES`:

- `"binary_nepc"` — one LLM call per patient; tight per-trigger context, no
  real per-note cap, one big per-patient payload budget (300k chars).
- `"longitudinal"` — used by cancer_stage, gleason_score, longitudinal_NEPC;
  wider per-trigger context (needed to date events), a real per-note cap, and
  a smaller per-chunk payload budget (60k chars) so patients with many notes
  get multiple LLM calls instead of one truncated call.

### Key env vars (all optional; sensible cluster defaults baked in)

| Env var | Default |
|---|---|
| `LLM_ANNOTATIONS_DATA_PATH` | `/data/gusev/USERS/jpconnor/data/LLM_annotations/` |
| `BINARY_NEPC_OUTPUT_DIR` | `<data_path>/LLM_NEPC_labels/` |
| `STAGE_OUTPUT_DIR` | `/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/` |
| `CAIA_AZURE_OPENAI_ENDPOINT` / `_API_VERSION` / `_MODEL` | DFCI Azure OpenAI endpoint / `2024-04-01-preview` / `gpt-4o` |
| `VERTEX_PROJECT` / `VERTEX_LOCATION` / `VERTEX_MODEL` | `gusevlabllm` / `us-central1` / `gemini-2.0-flash-001` |

The legacy `CAIA_COMPASS_*` env vars are still accepted as fallbacks for the
data-path variables.

### Authentication

- `dfci_gpt` — `DefaultAzureCredential` (AAD token); no API key needed, resolves
  automatically via Azure CLI login or managed identity in the DFCI environment.
- `vertex_ai` — Google Application Default Credentials.

### Note types

Notes are classified `Clinician`, `Imaging`, or `Pathology` from filename
patterns in the OncDRS source files. This drives both cleaning rules
(`preprocessing/utils.py`) and snippet-selection heuristics.

### Notebooks

`notebooks/<task>.ipynb` is the primary way to run a task end to end: set the
`PROVIDER` toggle, flip on the `RUN_*` cells you need, run top to bottom. Each
notebook subprocess-calls the relevant `preprocessing/cli/` script(s) and then
the task's runner with `--provider PROVIDER`.
