# CLAUDE.md

Guidance for Claude Code when working in this repository. All commands are run
from the repo root.

## Commands

```bash
# Install (core + one provider's SDK)
pip install -e ".[dfci_gpt]"     # or ".[vertex_ai]"

# Cancer stage / Gleason: collect evidence, then extract
python preprocessing/cli/extract_stage_notes.py --output-dir /path/to/out          # stage: scan (no LLM)
python tasks/cancer_stage/run_stage_extraction.py --output-dir /path/to/out --provider vertex_ai

python preprocessing/cli/collect_gleason_notes.py --output-dir /path/to/out
python tasks/gleason_score/build_gleason_timeline.py --output-dir /path/to/out --provider dfci_gpt

# Per-note NEPC / AVPC: collect trigger notes, then one LLM call per note
python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/out
python tasks/nepc/run_nepc_extraction.py --output-dir /path/to/out --provider vertex_ai
python preprocessing/cli/collect_avpc_notes.py --output-dir /path/to/out
python tasks/avpc/run_avpc_extraction.py --output-dir /path/to/out --provider vertex_ai
python tasks/avpc/run_avpc_extraction.py --output-dir /path/to/out --rebuild-labels-only  # re-gate, no LLM
python preprocessing/cli/collect_avpc_adjacent_notes.py --output-dir /path/to/out
python tasks/avpc_adjacent/run_avpc_adjacent_extraction.py --output-dir /path/to/out --provider vertex_ai
python preprocessing/cli/collect_arlpc_notes.py --output-dir /path/to/out    # Pathology + Clinician only
python tasks/arlpc/run_arlpc_extraction.py --output-dir /path/to/out --provider vertex_ai

# Disease course (BCR / progression / mCRPC): one shared per-note extraction,
# then three builders that make no LLM calls
python preprocessing/cli/discover_structured_codes.py      # confirm LABS / MEDICATIONS codes on the cluster
python preprocessing/cli/collect_course_notes.py --output-dir /path/to/out
python tasks/disease_course/run_course_extraction.py --output-dir /path/to/out --provider vertex_ai
python tasks/bcr/build_bcr_labels.py --output-dir /path/to/out
python tasks/progression/build_progression_events.py --output-dir /path/to/out
python tasks/mcrpc/build_mcrpc_labels.py --output-dir /path/to/out --met-source avpc   # or met_dx

# Collectors cache their evidence: re-running with the same
# scan settings reuses it, changed settings raise until you pass --overwrite.
python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/out --scan-workers 16
python preprocessing/cli/collect_nepc_dx_notes.py --output-dir /path/to/out --context-chars 4000 --overwrite

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
   and `collect_gleason_notes.py`/`collect_nepc_dx_notes.py` use
   `ProcessPoolExecutor` for the per-note scan (`--scan-workers`, default
   `os.cpu_count()`); dedup and chunk packing stay single-process because dedup
   must see the whole cohort. `extract_stage_notes.py` lazily prefilters the
   PROFILE_DATA parquets to stage-bearing notes before materializing them.

   All evidence-producing collectors write a metadata sidecar or bundle metadata
   recording a hash of the resolved scan settings. Re-running with unchanged
   settings reuses the existing evidence and skips the scan; changed settings
   raise rather than silently mixing incompatible evidence, so `--overwrite` is
   required to rescan.
2. **Task runner** (`tasks/<task>/`) — reads the evidence artifact and calls
   the selected provider. Stage and Gleason pack per-patient evidence into
   bounded chunks; the per-note tasks use the shared `tasks/note_extraction.py`
   runner. Evidence, model, prompt, payload sizing, and output schema are
   fingerprinted to prevent incompatible resume.

   Gleason resumes per chunk through `gleason_processed_chunks.parquet`: only
   failed/missing chunks are rerun. The evidence `scan_config` is checked before
   reuse. Per-note tasks resume through their processed-note ledgers and support
   `--retry-failures`.

   Active AVPC and disease-course builders share the deterministic disease-state
   gates in `tasks/mcrpc_gate.py` and `tasks/met_veto.py`. The metastatic gate
   separates M1 from N1; regional pelvic nodal disease alone does not qualify.
   Per-note label metadata records gate versions, and a label-only rebuild
   reapplies changed gates without new LLM calls.

Patient chunking is lossless: patients with many notes get multiple LLM calls
rather than truncation, so rare findings are never silently dropped.

### Per-note pipelines (`tasks/nepc`, `tasks/avpc`)

These are a separate design from the chunked pipelines above, and are
unchanged by them.

**Collectors.** `preprocessing/evidence.py::collect_note_evidence` writes one
evidence row per deduplicated trigger note. There is no chunk packing. It
optionally takes a per-note-type context override: NEPC uses
`{"Pathology": None}` to keep whole pathology reports. That override is hashed
into the scan config only when it is set, so existing scan hashes are
unchanged. Reports longer than the per-note cap retain the header and every
trigger passage with reduced context instead of truncating their beginning.
Both per-note collectors preserve eligibility headings and clinical reset
headings around snippets. AVPC/NEPC evidence v2 invalidate earlier evidence
that omitted that context; NEPC also fixes marker spellings and late addenda.

**Runner.** `tasks/note_extraction.py` is the shared runner, configured by a
`TaskSpec`.

- It makes one call per note, with one correction retry on unparseable output.
- It grounds each quote against that note.
- It keeps a per-note ledger (`<prefix>_processed_notes.parquet`), compacted to
  each note's final attempt.
- It persists a run fingerprint covering evidence, model, prompt, and finding
  schema.

**Labels.** Labels are built at rollup time from the stored findings.

- `tasks/nepc/ne_gate.py` screens NE quotes. The negation window is
  sentence-bounded. IHC polarity is read per marker, and a "positive for" or
  "negative for" cue binds only forward.
- `tasks/avpc/criteria.py` applies Aparicio thresholds, pairings, and C7
  interval rules. Every number is re-checked against its quote.
- The disease-state gates live in shared `tasks/mcrpc_gate.py` and `tasks/met_veto.py`.
- `tasks/trial_guard.py` rejects trial/eligibility text for every finding.
  At label-build time, the runner also locates quotes in their source evidence
  to detect cropped text from eligibility sections; a separate clinical
  assessment remains usable. Outside those sections a trial cue blocks only
  its own clause. Its regex (`TRIAL_CUE`) is deliberately narrower than
  `met_veto.BOILERPLATE`, which also matches clinical uses such as "per CAP
  protocol" or "prior study of". These transient context reasons do not change
  the stored extraction rows and are recomputed on label-only rebuilds.
- Gate and criteria versions go into `<prefix>_labels.meta.parquet`, not into
  the extraction fingerprint. So `--rebuild-labels-only` re-applies them
  without LLM calls. The metadata also records `trial_guard_version`.

**NEPC pathology groups.** `tasks/nepc/tiers.py` derives four groups:

- **t1:** definitive small-cell, large-cell NE or high-grade NE carcinoma,
  including mixed tumors with a confirmed NE carcinoma component. IHC alone
  and transformation without a definitive carcinoma target do not qualify.
- **t2:** explicitly poorly differentiated/dedifferentiated carcinoma plus NE
  features or positive NE tissue IHC. Gleason grade alone does not qualify.
- **t3:** explicit prostate adenocarcinoma plus NE features or positive NE IHC.
  Gleason 4+5=9 with focal weak synaptophysin staining is t3, not t1/t2.
- **t4:** documented prostate adenocarcinoma, completed review, and no gated NE
  evidence in the patient's extracted history. Cohort membership supplies
  clinical eligibility. Clinical aggressiveness alone does not exclude t4.

Pathology reports and explicit pathology summaries in clinician notes count;
bare assessments and IHC without histology do not establish a group. Pairing
is within one note, rejects explicitly different specimen sites, and uses
`onset.both_hold` for dates. `nepc_t*_supporting_quotes` retains both source
passages for a combined finding. `nepc_tier` selects t1 > t2 > t3 > t4;
`unclassified` means reviewed but insufficient evidence. Partial/failed/unrun
review cannot establish a reference negative; unsupported flags remain null.

`nepc_status` retains its legacy broad NE-evidence meaning and
`has_stated_nepc_dx` still accepts clinical assertions. Neither is the
pathology-defined definitive group. `primary_label` continues to use broad
`nepc_status` for compatibility; `primary_label_definitive` keys on `nepc_t1`.

Evidence v3 adds adenocarcinoma/histology triggers; prompt v4 adds
`prostate_adenocarcinoma` and requests pathology context. Migration requires
collector and extraction `--overwrite` (collector overwrite discards prior
outputs), not just a label rebuild. Tier v2 and NE gate v5 are recorded in
label metadata. NE gate v5 also accepts explicitly asserted focal/weak NE
staining; AVPC C6 shares this tissue-marker interpretation.

NE gate v6 (label-only rebuild) closes hedge leaks found in review. Before the
term it adds "possibility", "question of", "look for", "hypothesis" and
"doubtful". After the term it adds bare "(less/more/most) likely",
"unlikely" and "doubtful". A "?" attached to the term ("?NEPC", "NEPC?",
"NE features? Awaiting stains") also hedges it. The post-term "likely" also
applies to `screen_asserted` callers in AVPC and adjacent. `screen_any_asserted`
passes when any one occurrence is asserted; the visceral malignancy-word check
uses it.

**Review policy v2** (`note_extraction.review_incomplete`, label-only rebuild).
A `partial:K/N` review is treated like failed/not_run in all four per-note
tasks (NEPC, AVPC, AVPC-adjacent, ARLPC): positives stand, negatives stay
null. `review_policy_version` is recorded in each `labels.meta`. The disease
course builders keep their own `UNLABELED_STATUSES`.

**C2/C3 numbering.** `tasks/avpc` uses the paper's C2/C3 numbering: C2 =
exclusively visceral, C3 = lytic.

AVPC labels also carry `metastatic_date_precision` / `_source` and
`crpc_date_precision` / `_source` (criteria v3, additive; label-only rebuild).

Criteria v4 (label-only rebuild) tightens four criteria after manual review:

- **C3:** one lytic lesion is not predominance. The patient's other bone
  descriptions within 180 days veto it: any blastic, sclerotic or mixed wording
  vetoes plural-lytic wording, and "predominantly / diffusely sclerotic" vetoes
  even "predominantly lytic".
- **C4:** the measurement must belong to the node or the prostate/pelvic mass,
  so "CT abdomen/pelvis: 6 cm liver mass" does not count.
- **C5:** the PSA and the bone burden must be dated certainly within 90 days.
  - A pre-ADT PSA cannot follow the first ADT start. A note-dated diagnostic
    PSA retold in a later note is undated.
  - A castrate-setting PSA cannot precede the first ADT start.
  - ≥20 needs a written lesion count or innumerable / TNTC / superscan.
  - `c5_psa_date`, `c5_bone_date`, `c5_psa_context` and `c5_gap_days` are
    audit columns.
- **C7:** the anchor is the earliest first-line ADT start. Its date must be
  stated to month precision, and no CRPC may be documented before it. An
  ARPI-only start or "progressed within N months of enzalutamide" never
  anchors.

Criteria v5 (label-only rebuild):

- **Visceral:** at least one malignancy word (metastatic / mets / malignant /
  carcinoma) must itself be asserted. "consistent with" / "c/w" are no longer
  malignancy words, so "adrenal nodule consistent with adenoma" and "not
  consistent with metastases" fail. This also keeps such rows out of
  `is_metastatic`. Adjacent criteria v4 applies the same rule to A1.
- **Gleason:** a total needs Gleason/GS context or an `a+b` sum. A bare `= 8`
  ("PSA = 8") no longer matches.

**AR-low / AR-indifferent (`tasks/arlpc`).** A third per-note `TaskSpec` run on
the same runner. The collector scans Pathology (whole reports) and Clinician
notes only; Imaging is skipped because PSMA-PET never counts. Labels are
binary, `arlpc_`-prefixed, and independent of NE status and disease state, in
two tiers:

- **strict:** AR-low tissue IHC (negative, or any reduced wording such as weak,
  focal, patchy or reduced vs prior), AR-specific statements
  (AR-null/low/indifferent, double-negative), and somatic genomic AR loss.
- **broad:** strict plus androgen-independent / hormone-refractory wording.

Each tier records its `onset.earliest` date, and `arlpc_tissue_date` records
the earliest AR-low IHC date. AR-target IHC loss (PSA/PSAP/NKX3.1/PSMA/HOXB13)
is corroborating columns only, and AR-positive IHC is an audit count only.

`tasks/arlpc/ar_gate.py` has these rules:

- Wording decides moderate/strong versus reduced; percent and H-score are
  stored only.
- A quote or model that favors a non-prostate origin vetoes the finding.
- AR-V7, amplification, mutation and germline results are rejected.
- It shares `tasks/ihc_polarity.py` with `ne_gate`, but an equidistant
  positive/negative cue tie is unresolved for AR, not negative.

The gate's version goes in `labels.meta` as `ar_gate_version`. The
`primary_subtype` notebook left-joins these columns as annotation only, and
`primary_label` precedence is unchanged.

**AVPC-adjacent (`tasks/avpc_adjacent`).** A fourth per-note `TaskSpec`, fully
separate from `tasks/avpc` (own collector, prompt, ledger and labels; it never
reads `avpc_labels`). Its prompt is a deliberate copy of the AVPC prompt, not an
import, so the AVPC fingerprint is unchanged. It adds three things: an extended
`visceral_metastasis`, a `lytic_component` field on `bone_metastasis`, and an
`aggressive_variant_statement` item. There is no disease-state gate. A patient
is `is_avpc_adjacent` if any of these hold:

- **A1:** any visceral or soft-tissue metastasis. Lymph nodes, bone-lesion
  soft-tissue/epidural extension (`bone_extension_not_visceral`) and T4 bladder
  or rectal invasion (`local_invasion_not_metastasis`) are vetoed.
- **A2:** any lytic component, including mixed lytic/blastic.
- **A3:** modified C5. It reuses `avpc.criteria.evaluate_c5` through its
  `is_high_volume=` hook, with extent wording such as extensive, diffuse or
  multiple.
- **A4:** an asserted AVPC / anaplastic / variant-CRPC statement. Anaplastic
  tumors of other primaries are rejected.
- **C1–C7:** any strict criterion, from `avpc.criteria.evaluate_patient` with
  its mCRPC gate ignored.

Label columns are `avpc_adj_`-prefixed except `is_avpc_adjacent` and
`avpc_adjacent_date`. `labels.meta` records `adjacent_criteria_version` plus the
AVPC criteria and gate versions.

### Disease course (`tasks/disease_course`, `bcr`, `progression`, `mcrpc`)

**Shared extraction.** `tasks/disease_course/run_course_extraction.py` is a
per-note `TaskSpec` run over `course_evidence.parquet`, using the same runner
as nepc/avpc. Its `max_findings_per_note` is 120; that field is hashed only
when it differs from the default, so nepc/avpc fingerprints are unchanged.

It extracts atomic findings:

- PSA values (`=`, `<`, `>` or `undetectable`)
- RP and RT
- therapy start, current and stop
- BCR, progression, CRPC and mCRPC statements
- imaging recurrence

It writes `course_labels.parquet`, which holds the RP and primary/salvage RT
index dates.

**Builders.** The three builders make no LLM calls. They read the findings
through `note_extraction.load_label_inputs` and `disease_course/statement_gate.py`,
and drop every trial-context row before any rule runs. They combine those
findings with structured `preprocessing/structured.py` loaders: LABS PSA and
testosterone, MEDICATIONS, and the vital-status registry. Their outputs:

- `bcr_labels`
- `progression_events` plus `progression_labels`
- `mcrpc_labels`

Each output has a `.meta.parquet` that records:

- versions
- the course run config
- structured-source path/size/mtime
- every rule default

Re-running a builder after a gate or rule change needs no extraction rerun.

**PSA rules** (`disease_course/psa.py`):

- **Merge.** LABS wins for the same draw. A note value matching a LABS draw
  within the prior 7 days at the note's precision is a restatement, never a
  separate draw.
- **AUA.** Requires observed clearance (<0.1, or undetectable), then ≥0.2
  confirmed by a later separate draw.
- **Phoenix.** The preceding nadir + 2.
- **PCWG3.** 30-day pretreatment baseline, ≥25% and ≥2 rise, onset ≥12 weeks,
  confirmation ≥3 weeks later in the same therapy episode.

**Therapy coverage** (`disease_course/therapy.py`) is tiered
supported / reported / estimated:

- **Supported:** repeated doses (median gap), a parseable formulation, or
  days_supply.
- **Reported:** current/stop statements.
- **Estimated:** a lone record of unknown formulation. Estimated coverage
  alone never establishes castration.

**mCRPC** (`tasks/mcrpc/criteria.py`) is the earlier of:

- the date both metastatic disease and CRPC hold. Metastatic disease comes
  from `--met-source avpc|met_dx`, and the upstream label's sha256 and
  metadata are recorded.
- a dated mCRPC statement.

Derived CRPC requires one of:

- measured castration: testosterone <50 within ±60 days
- assumed castration: supported/reported ADT at both the rise and the
  confirmation

Testosterone ≥50 vetoes derived CRPC. A cabazitaxel / radium-223 / sipuleucel-T
start is a separate `drug_proxy` basis. It fills a missing date, or caps an
earlier one with `proxy_precedes_primary=True`. Lu-177-PSMA is audit-only.

Progression events have these properties:

- They carry the therapy exposure on their date.
- For failed or not-run patients they hold only the registry death.
- `events.first_event_after` picks a PFS event downstream.

All repository-owned persisted I/O uses Zstandard-compressed Parquet, including
evidence, snippet bundles, run/scan metadata, processed ledgers, rejected or
failed rows, raw findings, and final timelines/labels. JSON is restricted to
provider request/response payloads and serialized audit values inside Parquet.
Externally supplied MRN cohort lists are CSV inputs and are not pipeline artifacts.
The prostate-specific collectors default to
`$COMPASS_PATH/mrn_lists/icd_prostate_mrn_flags.csv`; cancer stage
remains pan-cancer unless an MRN restriction is supplied.

### Snippet sizing

`preprocessing/config.py` defines `SnippetProfile(context_chars, max_chars,
payload_max_chars)` and `SNIPPET_PROFILES`:

- `"patient_snippets"` — reusable patient snippet bundles with tight context
  and a 300k-character per-patient budget; optional preprocessing for review.
- `"longitudinal"` — used by cancer_stage, gleason_score, and per-note collectors;
  wider per-trigger context (needed to date events), a real per-note cap, and
  a smaller per-chunk payload budget (60k chars) so patients with many notes
  get multiple LLM calls instead of one truncated call.

### Key env vars (all optional; sensible cluster defaults baked in)

| Env var | Default |
|---|---|
| `LLM_ANNOTATIONS_DATA_PATH` | `/data/gusev/USERS/jpconnor/data/LLM_annotations/` |
| `PROFILE_DATA_PATH` | `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/` |
| `COMPASS_PATH` | `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/` |
| `STAGE_OUTPUT_DIR` | `/data/gusev/USERS/jpconnor/data/LLM_stage_extraction/` |
| `CAIA_AZURE_OPENAI_ENDPOINT` / `_API_VERSION` / `_MODEL` | DFCI Azure OpenAI endpoint / `2024-04-01-preview` / `gpt-4o` |
| `VERTEX_PROJECT` / `VERTEX_LOCATION` / `VERTEX_MODEL` | `gusevlabllm` / `us` (multi-region; Gemini 3+ isn't served in single US regions) / `gemini-3.5-flash-lite` |

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
the task's runner with `--provider PROVIDER`. `disease_course.ipynb` additionally runs the three
builders (`RUN_BUILDERS`). The exception is
`primary_subtype.ipynb`, which makes no LLM calls. Its logic lives in
`tasks/primary_subtype.py`. It full-joins `nepc_labels`, `avpc_labels` and
`avpc_adjacent_labels` and adds three columns:

- `primary_label` (`nepc` > `avpc` > `avpc_adjacent` > `prad`; `prad` was
  formerly `conventional`), keyed on broad `nepc_status`.
- `primary_label_definitive`, the same cascade keyed on `nepc_t1`.
- `small_cell_discordant`, which flags AVPC/adjacent C1 small-cell findings
  that NEPC did not label `nepc`. It is diagnostic only.

Each label is null whenever a higher-precedence label is unresolved. Each
input's `labels.meta` must match the current gate/criteria/review-policy
versions, otherwise it raises with a `--rebuild-labels-only` hint.
`primary_subtype_labels.meta.parquet` records each input's path, sha256 and
label meta.
