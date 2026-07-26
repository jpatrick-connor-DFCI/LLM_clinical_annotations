# LLM clinical annotations

LLM-based extraction of structured clinical annotations (NEPC status, cancer
stage, Gleason score, AVPC/NEPC criteria timelines) from prostate cancer
clinical notes, runnable against either DFCI Azure OpenAI or Google Vertex AI
(Gemini) as the LLM backend.

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

## Running a task

The easiest entry point is the matching notebook in `notebooks/`: set the
`PROVIDER` toggle (`"dfci_gpt"` or `"vertex_ai"`), flip on the `RUN_*` cells you
need, and run top to bottom.

To run from the command line instead:

```bash
# 1. Preprocessing (provider-independent)
python preprocessing/cli/compile_prostate_notes.py --output-path /path/to/notes.csv
python preprocessing/cli/compile_patient_snippets.py --output-path /path/to/snippets.json.gz

# 2. Task runner (provider-flagged)
python tasks/binary_NEPC/run_NEPC_classifier.py \
    --snippets-path /path/to/snippets.json.gz \
    --provider dfci_gpt   # or vertex_ai
```

Every preprocessing CLI and task runner supports `--help`.

## Setup

```bash
pip install -e .                    # core deps (preprocessing/, providers/, tasks/)
pip install -e ".[dfci_gpt]"        # + openai, azure-identity
pip install -e ".[vertex_ai]"       # + google-genai
```

`dfci_gpt` authenticates via `DefaultAzureCredential` (Azure AD). `vertex_ai`
authenticates via Google Application Default Credentials and reads
`VERTEX_PROJECT` / `VERTEX_LOCATION` from the environment.
