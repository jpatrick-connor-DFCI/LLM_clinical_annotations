# CLAUDE.md

Guidance for Claude Code when working in this repository. All commands are run
from the repo root.

## Commands

```bash
# Install (core + one provider's SDK)
pip install -e ".[dfci_gpt]"     # or ".[vertex_ai]"

# Build the shared note source (required before most pipelines)
python preprocessing/cli/compile_patient_snippets.py

# Binary NEPC: compile snippets, then classify
python preprocessing/cli/compile_patient_snippets.py --output-path /path/to/patient_snippets.parquet
python tasks/binary_NEPC/run_NEPC_classifier.py --snippets-path /path/to/patient_snippets.parquet --output-dir /path/to/out --provider dfci_gpt

# Cancer stage / Gleason / longitudinal NEPC: collect evidence, then extract
python preprocessing/cli/extract_stage_notes.py --output-dir /path/to/out          # stage: scan (no LLM)
python tasks/cancer_stage/run_stage_extraction.py --output-dir /path/to/out --provider vertex_ai

python preprocessing/cli/collect_gleason_notes.py --output-dir /path/to/out
python tasks/gleason_score/build_gleason_timeline.py --output-dir /path/to/out --provider dfci_gpt

python preprocessing/cli/collect_nepc_notes.py --output-dir /path/to/out
python tasks/longitudinal_NEPC/build_nepc_timeline.py --output-dir /path/to/out --provider dfci_gpt

# Strict NEPC diagnosis date only (precision-biased variant of the above)
python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/out
python tasks/nepc_diagnosis/build_nepc_dx_labels.py --output-dir /path/to/out --provider dfci_gpt

# The two longitudinal collectors cache their evidence: re-running with the same
# scan settings reuses it, changed settings raise until you pass --overwrite.
python preprocessing/cli/collect_nepc_notes.py --output-dir /path/to/out --scan-workers 16
python preprocessing/cli/collect_nepc_notes.py --output-dir /path/to/out --context-chars 4000 --overwrite

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
PROFILE_DATA/CLINICAL_NOTES/*.parquet → task-specific evidence/snippet artifacts
                                          │
                          preprocessing.notes.load_notes()
                 (explicit note parquets > Parquet bundle > default parquets)
                                          │
                       preprocessing.utils.clean_note()
                                          │
                preprocessing.triggers: trigger regex scan → snippet extraction
                                          │
              preprocessing.snippets / preprocessing.longitudinal: patient chunking
                                          │
                    tasks/<task>/*.py: LLM calls via providers.get_provider(...)
                                          │
                 incremental Parquet writes → final dedup / timeline build
```

Direct parquet runs push MRN, note-type, and task candidate predicates into the
lazy scan. Python cleaning/process pools operate only on candidate notes. The
PROFILE source contract is the physical schema emitted by
`PROFILE_data_processing`: pathology/imaging rows contain `RPT_ID`, `DFCI_MRN`,
`EVENT_DATE`, `PROC_DESC`, `RPT_TYPE`, `RPT_TEXT`, and `FILE`; progress rows
contain `RPT_ID`, `DFCI_MRN`, `EVENT_DATE`, `INP_RPT_TYPE`, `PROVIDER_TYPE`,
`ENCOUNTER_TYPE_DESC`, `RPT_TEXT`, and `FILE`. `RPT_TEXT` is already the merged
complete text; do not append or require `NARRATIVE_TEXT` downstream.

### Two-phase task pattern

Every task is split into a **preprocessing** step and a **task runner**:

1. **Preprocessing** (`preprocessing/cli/`) — provider-independent. Regex
   trigger matching across notes, context-window extraction, writes an
   Parquet evidence/snippet artifact. `compile_patient_snippets.py`
   and `collect_gleason_notes.py`/`collect_nepc_notes.py` use
   `ProcessPoolExecutor` for the per-note scan (`--scan-workers`, default
   `os.cpu_count()`); dedup and chunk packing stay single-process because dedup
   must see the whole cohort. `extract_stage_notes.py` lazily prefilters the
   PROFILE_DATA parquets to stage-bearing notes before materializing them.

   All evidence-producing collectors write a metadata sidecar or bundle metadata
   recording a hash of the resolved scan settings. Re-running with unchanged
   settings reuses the existing evidence and skips the scan; changed settings
   raise rather than silently mixing incompatible evidence, so `--overwrite` is
   required to rescan.
2. **Task runner** (`tasks/<task>/`) — reads the evidence artifact, groups
   snippets into per-patient chunks (greedy packing up to `payload_max_chars`),
   calls the selected provider once per chunk via `ThreadPoolExecutor`, writes
   raw findings + a processed-patient log incrementally. `run_NEPC_classifier.py`
   resumes at **patient** granularity and supports `--retry-failures` to retry
   only prior failures. Staging and binary NEPC also persist a run fingerprint;
   changed evidence, model, prompt, payload sizing, or output schema requires
   `--overwrite` instead of silently mixing output generations.

   The two longitudinal timeline builders resume at **chunk** granularity via
   `avpc_nepc_processed_chunks.parquet` / `gleason_processed_chunks.parquet`, but
   they differ in how far a gap propagates. `gleason_score` resumes strictly per
   chunk: a patient whose chunk 2 failed re-runs only chunk 2, keeping the
   findings the other chunks already produced. `longitudinal_NEPC` instead
   compiles a patient history forward across its map calls — each chunk sees a
   deterministic digest of prior chunks' validated findings plus a short
   LLM-written narrative (`tasks/longitudinal_NEPC/history.py`), so grounding
   still only accepts quotes from that chunk's own notes. Because that history
   depends on a consistent forward pass, its resume re-runs from the **first**
   outstanding chunk of a patient onward, not just the gap: a patient whose
   chunk 2 failed re-runs chunks 2 and every later chunk. Both runners derive
   per-patient status as `ok` / `partial:N/M` / `failed:<err>`, and both record
   the evidence `scan_config` hash on each chunk row; a mismatch against the
   evidence sidecar raises instead of resuming onto chunks that no longer mean
   the same thing. `longitudinal_NEPC`'s run fingerprint also includes
   `HISTORY_VERSION`, so changing how carried history is built or presented
   forces `--overwrite` rather than mixing chunks produced under two different
   history contracts.

   `nepc_diagnosis` is the strict-precision variant of `longitudinal_NEPC`,
   answering only "does the record state an NEPC diagnosis, and when?". It
   shares the collector, chunking, and resume machinery but differs in four
   ways: a narrowed trigger set (the `avpc`/`avpc_atomic` families are dropped);
   a deterministic negation/hedge/assertion gate applied to every grounded quote
   (`tasks/nepc_diagnosis/veto.py`), which the model cannot override; quote
   grounding that requires source-date **equality** and rejects a provenance
   mismatch rather than silently re-dating the finding to another note; and a
   per-patient adjudication call producing one label row instead of a
   multi-event timeline. Its run fingerprint includes `VETO_VERSION`, so
   changing the gate requires `--overwrite` rather than mixing labels
   adjudicated under two different gates. `nepc_dx_rejected_findings.parquet`
   records every rejected finding with its reason and is the tuning signal for
   the gate.

Patient chunking is lossless: patients with many notes get multiple LLM calls
rather than truncation, so rare findings are never silently dropped.

All repository-owned persisted I/O uses Zstandard-compressed Parquet, including
evidence, snippet bundles, run/scan metadata, processed ledgers, rejected or
failed rows, raw findings, and final timelines/labels. JSON is restricted to
provider request/response payloads and serialized audit values inside Parquet.
Externally supplied MRN cohort lists are CSV inputs and are not pipeline artifacts.
The prostate-specific collectors default to
`$COMPASS_PATH/mrn_lists/adt_mrns.csv`; cancer stage
remains pan-cancer unless an MRN restriction is supplied.

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
| `PROFILE_DATA_PATH` | `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/` |
| `COMPASS_PATH` | `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/` |
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

Notes are classified `Clinician`, `Imaging`, or `Pathology` from the three
PROFILE_DATA parquet basenames. This drives both cleaning rules
(`preprocessing/utils.py`) and snippet-selection heuristics.

### Notebooks

`notebooks/<task>.ipynb` is the primary way to run a task end to end: set the
`PROVIDER` toggle, flip on the `RUN_*` cells you need, run top to bottom. Each
notebook subprocess-calls the relevant `preprocessing/cli/` script(s) and then
the task's runner with `--provider PROVIDER`.
