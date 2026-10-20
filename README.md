# LLM clinical annotations

LLM-based extraction of structured clinical annotations (prostate cancer
subtype, pan-cancer stage, Gleason score, AVPC/NEPC criteria timelines) from
merged PROFILE_DATA clinical-note parquets, runnable against either DFCI Azure
OpenAI or Google Vertex AI (Gemini) as the LLM backend.

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
  binary_subtype/        Patient-level prostate cancer subtype (NEPC / AVPC /
                         biomarker / conventional) classification.
  cancer_stage/          Cancer stage timeline extraction.
  gleason_score/         Gleason score / grade group timeline extraction.
  longitudinal_subtype/  AVPC (Aparicio criteria) / NEPC feature timelines.
  nepc/                  Per-note NEPC vs non-NEPC diagnosis, quote, and date.
  avpc/                  Per-note AVPC: metastatic, CRPC, and C1-C7 (paper
                         numbering), each with its own quote and date.
  note_extraction.py     Shared one-call-per-note runner used by nepc/ and avpc/.

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
python tasks/binary_subtype/run_NEPC_classifier.py \
    --snippets-path /path/to/snippets.parquet \
    --provider dfci_gpt   # or vertex_ai
```

Binary subtype deterministically repairs harmless schema drift (such as a scalar
returned instead of a one-item list or a grounded quote paired with the wrong
note date). Malformed, ungrounded, inconsistent, or truncated responses receive
up to two corrective calls before the patient is written to the failure
Parquet; configure this with `--max-output-correction-retries`.

Every preprocessing CLI and task runner supports `--help`.

The binary subtype, Gleason, longitudinal subtype, NEPC, and AVPC collectors are
prostate-specific. Their default cohort is
`$COMPASS_PATH/mrn_lists/icd_prostate_mrn_flags.csv`; `--mrns` or
`--mrn-file` can override it. Only the cancer-stage collector defaults to the
full pan-cancer cohort.

### Longitudinal subtype

The longitudinal subtype runner uses a map/reduce extraction of AVPC/NEPC
criteria:

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

python tasks/longitudinal_subtype/build_nepc_timeline.py \
    --output-dir /path/to/avpc_nepc \
    --provider vertex_ai
```

To regenerate the timeline after output-format changes without any LLM calls or
failure retries, use `build_nepc_timeline.py --rebuild-timeline-only` with the
same output directory, provider/model, and MRN cohort settings as the saved run.

### NEPC and AVPC (per-note pipelines)

Two standalone pipelines that sit alongside `binary_subtype` and
`longitudinal_subtype`. Both make **one LLM call per deduplicated trigger
note**. The model reports only atomic findings, each with a verbatim quote and
the date as written. Note date, type, and id come from the evidence row, never
from the model. Quotes must ground in their own note (minimum 8 characters).
Failures go to `<prefix>_rejected_findings.parquet` without losing the note's
other findings.

Labels use the stored findings and source evidence: negation/hedge gates,
numeric thresholds, and timing rules run in code at rollup time. So
`--rebuild-labels-only` re-applies a changed gate with no LLM calls.

Clinical-trial eligibility, inclusion/exclusion lists, protocol titles, and
generic participant requirements are screened for every finding, including
numeric AVPC components. The runner also checks the quote's location in its
source evidence: a short quote taken from an eligibility section is rejected
even if it omits the heading. Outside such sections, a trial mention blocks
only its own clause, and only research-specific wording counts ("trial",
"enrolled", "NCT…", "phase II", "protocol 21-123"). So "Liver biopsy showed
small cell carcinoma; enrolled on NCT0123" keeps its diagnosis, and ordinary
uses like "per CAP protocol" or "the prior study of 3/2021" veto nothing. The
NEPC prompt also names trial text as a common false flag and asks for quotes
taken from the patient's own diagnosis rather than the trial description.
Both collectors preserve eligibility headings and clinical section boundaries
when building snippets. A separate patient assessment in the same note remains
usable. These exclusions appear as `boilerplate:*` in `gate_rejections`, and
the guard version is recorded in the labels metadata.
Label-only rebuilds use the saved evidence to reapply this context check.

Dates keep the precision stated in the text (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`).
A stated date later than its own note is rejected. Resume is per note:
`--retry-failures` re-calls only notes whose last attempt failed.

```bash
# NEPC: nepc_status, nepc_quote, nepc_date (+ source/precision), and a strict
# has_stated_nepc_dx tier. Pathology reports are kept whole when they fit;
# overlong reports retain their header and trigger passages within the cap.
python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/nepc
python tasks/nepc/run_nepc_extraction.py --output-dir /path/to/nepc --provider vertex_ai

# AVPC: is_metastatic / is_castration_resistant / is_mcrpc, then cN_met,
# cN_quote, cN_date for C1-C7, component quotes for the composite criteria,
# and is_avpc / avpc_date.
python preprocessing/cli/collect_avpc_notes.py --output-dir /path/to/avpc
python tasks/avpc/run_avpc_extraction.py --output-dir /path/to/avpc --provider vertex_ai
```

**The new AVPC pipeline follows Aparicio et al. 2013's numbering**: C2 =
exclusively visceral, C3 = predominantly lytic bone. This is the reverse of
`binary_subtype` and `longitudinal_subtype`.

`is_avpc` requires metastatic disease, castration resistance, and at least one
criterion. C1 (small cell) is exempt from the castration-resistance
requirement. Metastatic status can be implied by C2, C3, or C5, and castration
resistance by C7.

`notebooks/primary_subtype.ipynb` merges the full NEPC and AVPC label tables
into `primary_subtype_labels.parquet`. It makes no LLM calls. It adds one
column, `primary_label`: `nepc` if NEPC is positive, else `avpc` if `is_avpc`,
else `conventional`. A patient is labelled `avpc` or `conventional` only when
every higher-precedence label resolved negative. If either pipeline failed, did
not run, or lacks the patient, `primary_label` stays null. The two columns both
tables share get a `nepc_` / `avpc_` prefix: `review_status` and
`gate_rejections`.

Before calling the LLM, the collectors print notes and characters per note
type. AVPC's triggers are intentionally broad, so check that summary to size
cost first.

In either notebook, setting `MRNS` restricts extraction to that pilot subset;
the cohort file still defines the full label denominator. AVPC and NEPC
evidence v2 retain trial section context; NEPC also adds supported marker
spellings and preserves late pathology addenda. Existing v1 evidence must be
recollected with `--overwrite` and then extracted again to include context,
notes, and passages omitted by the old collectors.

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
`$COMPASS_PATH/mrn_lists/icd_prostate_mrn_flags.csv`; externally
supplied overrides also use CSV with a `DFCI_MRN` column.

The native PROFILE text rows are consumed exactly as emitted by
`PROFILE_data_processing`. Pathology and imaging Parquets contain
`RPT_ID`, `DFCI_MRN`, `EVENT_DATE`, `PROC_DESC`, `RPT_TYPE`, `RPT_TEXT`, and
`FILE`; progress-note Parquets contain `RPT_ID`, `DFCI_MRN`, `EVENT_DATE`,
`INP_RPT_TYPE`, `PROVIDER_TYPE`, `ENCOUNTER_TYPE_DESC`, `RPT_TEXT`, and `FILE`.
Upstream processing has already merged `NARRATIVE_TEXT` into `RPT_TEXT`, so the
loader treats `RPT_TEXT` as the complete canonical note body.

Staging and binary subtype resume state is fingerprinted against
evidence/snippet content, provider, model, prompts, and output schemas. Binary
output uses `review_status` to distinguish `llm_classified`, `no_trigger`, and
`no_notes`.
