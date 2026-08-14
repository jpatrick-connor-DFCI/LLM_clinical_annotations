# LLM clinical annotations

LLM-based extraction of structured clinical annotations (NEPC status, pan-cancer
stage, Gleason score, AVPC/NEPC criteria timelines) from merged PROFILE_DATA
clinical-note parquets, runnable against either DFCI Azure OpenAI or Google
Vertex AI (Gemini) as the LLM backend.

## Layout

```text
preprocessing/     Provider-agnostic note loading, cleaning, trigger-scanning,
                    and snippet building. Never imports a provider SDK.
  cli/              Standalone preprocessing scripts (compile notes, collect
                    per-task evidence). Run before any LLM calls.
  bundles/          Snippet-bundle read/write helpers.

providers/          Thin adapters over each LLM backend behind one interface
                    (providers.base.Provider): build_client(), call_with_retry().
                    Each adapter lazily imports its own SDK only, so selecting
                    one provider never pulls in the other's dependencies.
  dfci_gpt.py        DFCI Azure OpenAI adapter.
  vertex_ai.py       Google Vertex AI (Gemini) adapter, via google-genai.

tasks/              One directory per extraction task, each with a system
                    prompt and a runner that takes --provider {dfci_gpt,vertex_ai}.
  binary_NEPC/       NEPC vs. adenocarcinoma classification.
  cancer_stage/      Cancer stage timeline extraction.
  gleason_score/     Gleason score / grade group timeline extraction.
  longitudinal_NEPC/ AVPC (Aparicio criteria) / NEPC feature timeline extraction.

notebooks/          One notebook per task with a PROVIDER toggle. Each notebook
                    subprocess-calls the preprocessing CLIs, then the task's
                    runner with --provider set from the toggle.
```

Each task follows the same two-phase pattern: a **preprocessing** step (provider-
independent — scans notes, writes an evidence/snippet artifact) followed by a
**task runner** (provider-flagged — reads that artifact, makes the LLM calls,
builds the output timeline/labels).

Compilation commands expose nested progress bars for overall steps, per-note
trigger scanning, evidence deduplication, patient ranking, and chunk packing.

Parquet scans apply cohort, note-type, and task-specific candidate predicates
before note text is materialized. Staging recognizes AJCC/base-stage, TNM, FIGO,
Rai, Binet, Durie-Salmon, and limited/extensive-stage language and preserves the
stated system/value plus a normalized I–IV group when applicable. Each staging
event also records explicit histology, primary site, and metastatic sites.

## Running a task

The easiest entry point is the matching notebook in `notebooks/`: set the
`PROVIDER` toggle (`"dfci_gpt"` or `"vertex_ai"`), flip on the `RUN_*` cells you
need, and run top to bottom.

To run from the command line instead:

```bash
# 1. Preprocessing (provider-independent). By default this reads
# PROFILE_DATA/CLINICAL_NOTES/{PATHOLOGY,IMAGING,PROGRESS}_NOTES.parquet.
python preprocessing/cli/compile_patient_snippets.py \
    --output-path /path/to/snippets.parquet

# 2. Task runner (provider-flagged)
python tasks/binary_NEPC/run_NEPC_classifier.py \
    --snippets-path /path/to/snippets.parquet \
    --provider dfci_gpt   # or vertex_ai
```

Binary NEPC deterministically repairs harmless schema drift (such as a scalar
returned instead of a one-item list or a grounded quote paired with the wrong
note date). Malformed, ungrounded, inconsistent, or truncated responses receive
up to two corrective calls before the patient is written to the failure
Parquet; configure this with `--max-output-correction-retries`.

Every preprocessing CLI and task runner supports `--help`.

The binary NEPC, Gleason, and longitudinal AVPC/NEPC collectors are
prostate-specific. Their default cohort is
`$COMPASS_PATH/mrn_lists/adt_mrns.csv`; `--mrns` or
`--mrn-file` can override it. Only the cancer-stage collector defaults to the
full pan-cancer cohort.

### Longitudinal AVPC/NEPC

The longitudinal AVPC/NEPC runner uses a map/reduce extraction:

1. `collect_nepc_notes.py` writes content-hashed evidence chunks.
2. `build_nepc_timeline.py` maps each chunk into validated atomic evidence.
3. A patient-level synthesis combines all chunk maps so composite Aparicio
   criteria can use facts documented in different notes/chunks.

The final timeline is cohort-complete when the MRN cohort is available. Patients
with no validated AVPC/NEPC criteria receive an undated `conventional` sentinel
row with zero cumulative criteria. Patients with failed or incomplete extraction
are not auto-labeled conventional.

Resume state is bound to the evidence content, provider, model, prompt text,
and output schema. If any of these change, rerun stage 1 and/or stage 2 with
`--overwrite` as instructed by the CLI rather than mixing incompatible runs.
Grounded items that fail validation are quarantined in
`avpc_nepc_rejected_findings.parquet`; affected successful rows use the
`ok_with_rejections` status so partial results remain visible and auditable.

```bash
python preprocessing/cli/collect_nepc_notes.py \
    --output-dir /path/to/avpc_nepc

python tasks/longitudinal_NEPC/build_nepc_timeline.py \
    --output-dir /path/to/avpc_nepc \
    --provider vertex_ai
```

To regenerate the timeline after output-format changes without any LLM calls or
failure retries, use `build_nepc_timeline.py --rebuild-timeline-only` with the
same output directory, provider/model, and MRN cohort settings as the saved run.

## Setup

```bash
pip install -e .                    # core deps (preprocessing/, providers/, tasks/)
pip install -e ".[dfci_gpt]"        # + openai, azure-identity
pip install -e ".[vertex_ai]"       # + google-genai
```

`dfci_gpt` authenticates via `DefaultAzureCredential` (Azure AD). `vertex_ai`
authenticates via Google Application Default Credentials and reads
`VERTEX_PROJECT` / `VERTEX_LOCATION` from the environment.

Set `PROFILE_DATA_PATH` to override the default
`/data/gusev/USERS/jpconnor/data/PROFILE_DATA/` root. Each preprocessing CLI
also accepts repeated `--notes-parquet` arguments for explicit file overrides.
Set `COMPASS_PATH` to override the default
`/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/` cohort root.
Raw clinical text and every pipeline-owned evidence, state, metadata, failure,
and result artifact are stored as Zstandard-compressed Parquet. JSON remains
only as the LLM wire format or as a value inside a Parquet audit column.
The default MRN cohort is
`$COMPASS_PATH/mrn_lists/adt_mrns.csv`; externally
supplied overrides also use CSV with a `DFCI_MRN` column.

The native PROFILE text rows are consumed exactly as emitted by
`PROFILE_data_processing`. Pathology and imaging Parquets contain
`RPT_ID`, `DFCI_MRN`, `EVENT_DATE`, `PROC_DESC`, `RPT_TYPE`, `RPT_TEXT`, and
`FILE`; progress-note Parquets contain `RPT_ID`, `DFCI_MRN`, `EVENT_DATE`,
`INP_RPT_TYPE`, `PROVIDER_TYPE`, `ENCOUNTER_TYPE_DESC`, `RPT_TEXT`, and `FILE`.
Upstream processing has already merged `NARRATIVE_TEXT` into `RPT_TEXT`, so the
loader treats `RPT_TEXT` as the complete canonical note body.

Staging and binary NEPC resume state is fingerprinted against evidence/snippet
content, provider, model, prompts, and output schemas. Binary output uses
`review_status` to distinguish `llm_classified`, `no_trigger`, and `no_notes`.
