# LLM clinical annotations

LLM-based extraction of structured clinical annotations (prostate cancer
subtype, pan-cancer stage, Gleason score, and disease-course events) from
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
  cancer_stage/          Cancer stage timeline extraction.
  gleason_score/         Gleason score / grade group timeline extraction.
  nepc/                  Per-note NEPC vs non-NEPC diagnosis, quote, and date.
  avpc/                  Per-note AVPC: metastatic, CRPC, and C1-C7 (paper
                         numbering), each with its own quote and date.
  avpc_adjacent/         Per-note AVPC-adjacent: relaxed, ungated criteria
                         A1-A4 plus strict C1-C7 without the mCRPC gate.
  disease_course/        Per-note prostate course extraction (PSA, local and
                         systemic therapy, recurrence/progression/CRPC
                         statements) plus shared PSA and therapy rules.
  bcr/                   BCR date and basis (AUA, Phoenix, statement, imaging).
  progression/           Dated progression events (PSA/PCWG3, radiographic,
                         clinical, death) for PFS.
  mcrpc/                 mCRPC diagnosis date.
  met_veto.py            Shared metastatic-disease quote gate.
  mcrpc_gate.py          Shared metastatic/castration-resistance quote gates.
  note_extraction.py     Shared one-call-per-note runner used by nepc/, avpc/,
                         avpc_adjacent/ and disease_course/.

notebooks/          One notebook per task with a PROVIDER toggle. Each notebook
                    subprocess-calls the preprocessing CLIs, then the task's
                    runner with --provider set from the toggle.
  prostate_subtype/   NEPC, AVPC, AVPC-adjacent, ARLPC, and primary subtype merge.
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
# Collect evidence, then extract findings and build patient labels.
python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/nepc
python tasks/nepc/run_nepc_extraction.py --output-dir /path/to/nepc --provider dfci_gpt
```

Every preprocessing CLI and task runner supports `--help`.

The Gleason, NEPC, AVPC, AVPC-adjacent, and ARLPC collectors are prostate-specific.
Their default cohort is `$COMPASS_PATH/mrn_lists/icd_prostate_mrn_flags.csv`;
`--mrns` or `--mrn-file` can override it. Only the cancer-stage collector defaults
to the full pan-cancer cohort.

### NEPC and AVPC (per-note pipelines)

These pipelines make **one LLM call per deduplicated trigger
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

# AVPC-adjacent: a1_visceral / a2_lytic / a3_c5mod / a4_statement plus ungated
# c1..c7 (all avpc_adj_-prefixed), then is_avpc_adjacent / avpc_adjacent_date.
python preprocessing/cli/collect_avpc_adjacent_notes.py --output-dir /path/to/avpc_adjacent
python tasks/avpc_adjacent/run_avpc_adjacent_extraction.py --output-dir /path/to/avpc_adjacent --provider vertex_ai
```

NEPC labels carry four pathology-based groups (`tasks/nepc/tiers.py`), each
with an earliest date, precision, evidence type, quote and supporting quotes:

| Column | Group |
| --- | --- |
| `nepc_t1` | Definitive NEPC: small-cell, large-cell NE, or high-grade NE carcinoma of prostatic origin, including mixed tumors with a confirmed NE carcinoma component. IHC alone does not qualify. |
| `nepc_t2` | Intermediate/indeterminate: explicitly poorly differentiated or dedifferentiated carcinoma with NE features or positive NE tissue markers, without a definitive NE carcinoma diagnosis. Gleason grade alone does not qualify. |
| `nepc_t3` | Explicit prostate adenocarcinoma with focal/diffuse NE features or positive NE tissue markers. Gleason 4+5=9 adenocarcinoma with focal weak synaptophysin staining belongs here. |
| `nepc_t4` | Conventional prostate adenocarcinoma: documented histology, completed review, and no gated NE evidence anywhere in the patient's extracted history. Existing cohort membership supplies clinical eligibility. Clinical aggressiveness alone does not exclude this reference group. |

Pathology reports and clinician notes explicitly summarizing a pathology result
qualify. Bare clinical assessments, generic NE features without established
histology, and isolated IHC remain broad evidence but cannot establish a group.
Adenocarcinoma and PD diagnoses can pair with NE evidence in the same note;
explicitly different specimens or non-overlapping biopsy dates do not pair,
and onset is when both components hold. Combined findings retain their source passages separately in
`nepc_t*_supporting_quotes`.

`nepc_tier` selects the highest group ever documented (`t1 > t2 > t3 > t4`).
The t1/t2/t3 flags may record different events over time; t4 is a patient-level
reference negative. Reviewed patients without sufficient histology are
`unclassified`, not t4. Unresolved flags and an unsupported group stay null for
failed, unrun or partially reviewed patients. Positive evidence can still set a
group during incomplete review.

The legacy `nepc_status` remains a **broad NE-evidence label**, including IHC;
`has_stated_nepc_dx` still includes clinical assertions. Use `nepc_t1` for
**definitive NEPC** and `nepc_t4` for reference negatives. The existing primary
subtype notebook still uses the broad `nepc_status` for its NEPC precedence.

The collector now includes adenocarcinoma and prostate-histology triggers to
retrieve conventional reports without NE terms. Migrating to evidence v3 and
prompt `nepc-note-v4` requires **recollection and re-extraction with
`--overwrite`**; the collector's overwrite discards previous task outputs.
Collection may select more notes. A label-only rebuild cannot recover omitted
histology. Subsequent rule-only changes can use `--rebuild-labels-only`;
metadata records `nepc_tier_version` and `ne_gate_version`.

Collection now deduplicates within each note type, preserving pathology
provenance even when an earlier clinician note has identical text. ARLPC
triggers also include heterogeneous, diminished, and other supported reduced
AR-expression descriptions. To apply these collection changes to existing
runs, recollect with `--overwrite`, run extraction again, then rerun the primary
subtype merge. Collector overwrite discards that task's previous outputs;
use a new output directory if you need to retain the previous run. A label-only
rebuild applies the revised gates and pairing rules to stored findings but
cannot recover omitted notes.

**The new AVPC pipeline follows Aparicio et al. 2013's numbering**: C2 =
exclusively visceral, C3 = predominantly lytic bone.

`is_avpc` requires metastatic disease, castration resistance, and at least one
criterion. C1 (small cell) is exempt from the castration-resistance
requirement. Metastatic status can be implied by C2, C3, or C5, and castration
resistance by C7.

C5 numeric lesion counts are attributed to bone rather than another metastatic
site in the same quote. AVPC composite criteria preserve unknown onset when a
prerequisite is undated, and retain the coarsest overlapping date precision.
Criterion `*_date_precision` fields and `avpc_date_precision` accompany the
dates; AVPC-adjacent likewise exports criterion precision and
`avpc_adjacent_date_precision`.

**AVPC-adjacent** (`tasks/avpc_adjacent`) is a separate, relaxed per-note
pipeline for patients who look aggressive-variant without meeting the strict
definition. It has its own collector, prompt, LLM run, ledger and labels, and
does not read the AVPC outputs. There is **no disease-state gate**: a patient
is `is_avpc_adjacent` when any of these is documented.

| Criterion | Definition |
| --- | --- |
| A1 | Any visceral or soft-tissue metastasis: the C2 organs plus dura / leptomeninges / spinal cord, discrete bladder or rectal metastases, and soft-tissue / subcutaneous / skin / muscle deposits. Lymph nodes, the soft-tissue or epidural extension of a bone lesion, and T4 local invasion do not count. |
| A2 | Any lytic bone metastasis, including mixed lytic/blastic (`lytic_component`). |
| A3 | Modified C5: low PSA as in C5 (≤10 ng/mL, pre-ADT or symptomatic-CRPC, within 90 days) plus extensive / widespread / diffuse / numerous / multiple / multifocal / high-volume / heavy-burden bone metastases, with no lesion count needed. |
| A4 | An explicit aggressive-variant / AVPC / anaplastic / variant-CRPC statement. Hedged ("concern for", "?AVPC"), negated, and trial text are rejected, as are androgen-indifferent wording alone and anaplastic tumors of other primaries. |
| C1-C7 | Any strict Aparicio criterion, recomputed from this run's own findings without the mCRPC requirement. |

`avpc_adjacent_date` is the earliest qualifying criterion date. All label
columns are `avpc_adj_`-prefixed except `is_avpc_adjacent` and
`avpc_adjacent_date`. `--rebuild-labels-only` works as for AVPC; the labels
metadata records `adjacent_criteria_version` alongside the strict versions.

`notebooks/prostate_subtype/primary_subtype.ipynb` merges the full NEPC, AVPC, and AVPC-adjacent
label tables into `primary_subtype_labels.parquet`. It makes no LLM calls; the
logic is in `tasks/primary_subtype.py`. It adds `primary_label`, in precedence
order NEPC > AVPC >
AVPC-adjacent > PRAD: `nepc` if NEPC is positive, else `avpc` if `is_avpc`,
else `avpc_adjacent` if `is_avpc_adjacent`, else `prad` (formerly
`conventional`). A patient gets a lower label only when every higher-precedence
label resolved negative. If any of the three pipelines failed, did not run,
only partly ran, or lacks the patient, `primary_label` stays null.
`primary_label_definitive` applies the same cascade to `nepc_t1` (definitive
pathology NEPC) instead of broad `nepc_status`. `small_cell_discordant` flags
patients whose AVPC C1 small-cell finding NEPC did not label, for manual
review. Stale input labels (gate or criteria versions older than the code)
raise; re-run that pipeline with `--rebuild-labels-only`. A
`primary_subtype_labels.meta.parquet` sidecar records each input's sha256. The
two columns the NEPC and AVPC tables share get a `nepc_` / `avpc_` prefix:
`review_status` and `gate_rejections`.

Before calling the LLM, the collectors print notes and characters per note
type. AVPC's triggers are intentionally broad, so check that summary to size
cost first.

In either notebook, setting `MRNS` restricts extraction to that pilot subset;
the cohort file still defines the full label denominator. AVPC and NEPC
evidence v2 retain trial section context; NEPC also adds supported marker
spellings and preserves late pathology addenda. Existing v1 evidence must be
recollected with `--overwrite` and then extracted again to include context,
notes, and passages omitted by the old collectors.

### Disease course: BCR, progression events, mCRPC date

One per-note extraction (`tasks/disease_course`, same runner as NEPC/AVPC)
records PSA values, RP/RT, therapy starts/stops, and recurrence, progression,
CRPC and mCRPC statements, each with its quote and date. Three builders then
combine those findings with structured LABS PSA and testosterone, MEDICATIONS,
and the vital-status registry. They make no LLM calls, so a rule change needs
only a builder rerun.

```bash
python preprocessing/cli/discover_structured_codes.py   # check LABS / medication codes first
python preprocessing/cli/collect_course_notes.py --output-dir /path/to/course
python tasks/disease_course/run_course_extraction.py --output-dir /path/to/course --provider vertex_ai
python tasks/bcr/build_bcr_labels.py --output-dir /path/to/course
python tasks/progression/build_progression_events.py --output-dir /path/to/course
python tasks/mcrpc/build_mcrpc_labels.py --output-dir /path/to/course   # needs avpc_labels (or --met-source met_dx)
```

- **`bcr_labels.parquet`**: `bcr_date` and `bcr_basis`. The bases are:
  - `psa_aua`: after RP. Clearance (<0.1, or undetectable) must be observed
    first, then ≥0.2 is confirmed by a later draw.
  - `psa_phoenix`: after primary RT, at the nadir + 2.
  - `statement`: a clinician's statement.
  - `imaging`: imaging recurrence.

  The file also has `persistent_psa`, the clearance date and per-basis audit
  columns. Persistent PSA is flagged but is not itself BCR.
- **`progression_events.parquet`**: one row per dated PSA (statement or
  PCWG3), radiographic, clinical or death event, with the therapy exposure on
  that date and how certain it is. `progression_labels.parquet` gives the first
  event of each type, plus the registry death and last-alive dates.
- **`mcrpc_labels.parquet`**: `mcrpc_date`, the earlier of:
  - the date metastatic disease (from the AVPC labels by default) and CRPC
    both hold
  - a dated mCRPC statement

  Derived CRPC requires measured castration (testosterone <50 near the rise) or
  documented ADT at both the rise and its confirmation. A start of
  cabazitaxel, radium-223 or sipuleucel-T is a separate `drug_proxy` basis. It
  fills a missing date, and caps an earlier one with
  `proxy_precedes_primary=True`. Lu-177-PSMA alone never counts.

Every output has a `.meta.parquet` with its rule versions, defaults and
structured-source identities. `notebooks/disease_course.ipynb` runs all of it.

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

Extraction resume state is fingerprinted against evidence content, provider,
model, prompts, and output schemas. Per-note pipelines preserve completion
status in their processed-note ledgers and patient labels.
