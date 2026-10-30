# Prostate disease-course endpoints: BCR, progression events, mCRPC date

## Context

Three new prostate-specific labels are needed, each dated and auditable:

1. **Biochemical recurrence (BCR)**: the date and the basis (PSA rule, clinician statement, or imaging).
2. **Progression events**: every dated PSA, radiographic, clinical and death event, with the therapy the patient was on. The PFS index date is chosen downstream.
3. **mCRPC diagnosis date**.

Decisions already made:
- **One shared per-note LLM extraction** (`tasks/disease_course`) on the existing `TaskSpec` runner.
- **Three builders that make no LLM calls.** They combine those findings with structured LABS, MEDICATIONS and the vital-status registry, plus the AVPC metastatic date.
- **Guideline-derived rules apply in code**: AUA, Phoenix and PCWG3. Clinician statements also count. Project-specific eligibility, timing and confirmation defaults are explicit below and recorded in label metadata.
- **LABS takes precedence for the same PSA measurement.** Note PSAs fill missing measurements, including distinct draws close in time to a LABS result.
- **mCRPC** = the first date where both "metastatic" and "CRPC" hold, or an earlier dated explicit "mCRPC" statement. An eligible drug start is a separately identified proxy: it supplies the date when neither primary basis is available, and caps the date (flagged for review) when it precedes the primary basis. Lu-177-PSMA alone is not eligible.

Defaults (all can be changed later with a label-only rebuild):
- The metastatic source is AVPC.
- Persistent PSA after RP is flagged but does not itself count as BCR. PSA-derived recurrence requires observed clearance followed by a qualifying new rise; missing clearance evidence is recorded as unevaluable.
- PSMA findings described as "suspicious" count as *probable*; "equivocal" findings do not count.
- Estimated medication coverage alone cannot establish castration. Missing testosterone may support an assumed-castrate tier only when active ADT has independent evidence.

Existing extraction behavior and fingerprints must not change. The only planned existing-label change is the additive AVPC date provenance columns below.

## Approach

### 1. Structured data loaders: new `preprocessing/structured.py` (no provider imports)

All loaders use a lazy `pl.scan_parquet` on `PROFILE_DATA_PATH` with the MRN filter pushed into the scan, mirroring `load_profile_notes` in `preprocessing/notes.py`.

**`load_psa(mrns)`**
- Codes: `PSA, PSAR, PSATOTSCRN, PSAMON, PSAULT, PSAT`. This is the parent repo's `BROAD_PSA_CODES` without `CPSA`, which is complexed PSA.
- `TEXT_RESULT` "<x" becomes `comparator="<"` with `value=x`; preserve ">" bounds as well. Bounds are not exact measurements.
- The `9999999` sentinel together with "<" means below detection: recover a numeric bound from `TEXT_RESULT` when present. Otherwise keep an unknown detection limit for audit, not a numeric PSA. Other sentinels are dropped.
- Plausible range: 0–10000.
- Output columns: `DFCI_MRN, date, date_precision, psa_ng_ml, comparator, source="labs"`, plus source-row identity and specimen/accession identity when available. Preserve evidence needed to distinguish separate draws and conflicting results.

**`load_testosterone(mrns)`**
- Codes: TES / TESTT / TOTTES / TTEST.
- nmol/L is converted with ×28.84 to give ng/dL. Plausible range: 0–2000.

**`load_medications(mrns)`**
- Maps each drug name to a class: `depot_adt` (injectable/implant GnRH agents, including degarelix), `oral_adt` (relugolix), `antiandrogen`, `arpi`, `taxane`, `radium`, `lu_psma` (regex: lutetium | vipivotide | pluvicto), `sipuleucel`, `parpi`, `platinum`.
- Name lists are copied from the parent repo's `compile_COMPASS_cohort_data.py` and `classify_adt_intent.py`, but relugolix is explicitly removed from the parent's broadly named `DEPOT_ADT_MEDS` set and classified as oral ADT.
- Preserve source-row identity and any available formulation, dose, route, days-supply, administration and stop fields. The parent medication contract guarantees only MRN, drug name and start date; missing exposure details remain unknown.

**`load_vital_status(mrns)`**
- `HYBRID_DEATH_DT` gives the death date and `DERIVED_LAST_ALIVE_DATE` gives the last-alive date.
- Dates are parsed with `preprocessing.notes.to_iso_date`.
- Parse before reducing rows. `last_alive_date` is the maximum valid last-alive date, independent of row order.
- Deduplicate identical death dates. For conflicting valid death dates, default to the maximum to match the parent pipeline, set `death_date_conflict=true`, and retain all candidates and sources for audit. Version this conflict policy.
- Preserve and flag a last-alive date after the selected death date; do not silently clamp either date. Expose these registry flags in progression labels for downstream censoring decisions.

**`structured_source_fingerprint()`**
- Records path, size and mtime of each source file; the files are too large to hash.

**New CLI: `preprocessing/cli/discover_structured_codes.py`**
- Prints `TEST_TYPE_CD`/`DESCR`/`UOM` counts and medication-name counts, so the code lists can be confirmed on the cluster.
- Also reports, per ADT drug, which exposure fields (formulation, dose, route, days supply, administration, stop) exist and how often they are filled in, plus the distribution of intervals between repeated administrations. Use this before fixing the exposure defaults in `therapy.py`.

### 2. Shared extraction: `disease_course`

**Trigger regex**
- Add `COURSE_TRIGGER_REGEX` in `preprocessing/triggers.py`. It must be Rust-regex compatible: no lookaround, and the `\b(?:…)\b` style used elsewhere in that file.
- Families:
  - PSA / prostate-specific antigen
  - local therapy: prostatectomy, RALP/RARP, radiation, IMRT/EBRT/SBRT, brachytherapy, proton, salvage/adjuvant
  - systemic therapy: ADT agents, ARPIs, taxanes, radium/xofigo, lutetium/pluvicto, sipuleucel/provenge, orchiectomy
  - recurrence: recurren\*, relapse, biochemical failure, BCR, Phoenix, nadir, `progress(?:ion|ed|ing|ive)` (never bare "progress"), rising PSA, doubling time, new lesions
  - imaging: PSMA, pylarify, DCFPyL, Ga-68, axumin, fluciclovine, choline, PET, bone scan
  - CRPC terms: (n)mCRPC, castration-resistant, castrate-resistant, hormone-refractory

**Config**
- Add `COURSE_EVIDENCE_SCHEMA_VERSION = "course-note-evidence-parquet-v1"` to `preprocessing/config.py`.

**Collector: `preprocessing/cli/collect_course_notes.py`**
- A clone of `collect_avpc_notes.py`: prefix `course`, context 2000, default output dir `LLM_disease_course`.

**Runner change: `tasks/note_extraction.py`** (backward compatible)
- Add `TaskSpec.max_findings_per_note` (default 40). `extract_note` uses it instead of the module constant.
- `run_fingerprint` hashes it **only when it is not the default**, so the existing nepc/avpc fingerprints are identical.
- Add a read-only `load_label_inputs(spec, output_dir, evidence_path=None)`.
  - It compacts in memory and returns `notes`, the trial-annotated `findings`, `status`, `cohort` and `run_config`.
  - `run()`'s label step and the external builders both use it.

**Prompt: `tasks/disease_course/prompts.py`** (`COURSE_PROMPT_VERSION = "course-note-v1"`)
- Short quotes; one table row per PSA.
- Fills the existing `DATE_QUOTE_SCHEMA` (stated date, quote, confidence).
- Imaging recurrence describes disease presence; a radiographic `progression_event` requires an explicit new/worsening finding or a progression assessment. Repeated stable findings are not new progression events.

Items and their typed fields:

| item | fields |
|---|---|
| `psa_value` | `psa_ng_ml` (nullable), `value_comparator` (`=`/`<`/`>`/`undetectable`), `value_timing` (`dated`/`current`) |
| `radical_prostatectomy` | — |
| `radiation_therapy` | `rt_intent` (primary/adjuvant/salvage/palliative/unknown), `rt_target` |
| `therapy_start` | `agent` (stated date required) |
| `therapy_current` | `agent` |
| `therapy_stop` | `agent`, `stop_reason` (completed/toxicity/progression/holiday/other) |
| `bcr_statement` | `bcr_kind` (recurrence/persistent_psa) |
| `imaging_recurrence` | `modality`, `site_category` (local/nodal_pelvic/nodal_distant/bone/visceral), `certainty` (definite/probable/equivocal) |
| `progression_event` | `progression_type` (psa/radiographic/clinical/unspecified), `therapy_stated` |
| `crpc_statement` | — |
| `mcrpc_statement` | — |

**Runner: `tasks/disease_course/run_course_extraction.py`**
- `validate_finding` follows the `tasks/avpc/run_avpc_extraction.py` pattern: `ITEM_FIELDS`, the `_as_float`/`_as_text` helpers, and unknown enum values become None.
- Rules that prevent mis-dating:
  - `psa_value` requires a stated date or `value_timing="current"`. The number is re-checked with `tasks/avpc/criteria.number_in_quote`.
  - `value_comparator="undetectable"` captures "PSA undetectable" with no number. `psa_ng_ml` must be null, and the quote must contain an undetectable cue ("undetectable", "not detected", "<LLN"/"below detection"); a number in the same quote becomes a `<` bound instead.
  - `therapy_start` requires a stated date.
  - RP, RT and therapy-start quotes that match a planning cue (the `ADT_PLANNED` vocabulary) are rejected.
- `max_findings_per_note=120`.
- `build_labels` writes `course_labels.parquet`, with one row per cohort MRN:
  - `rp_date`, `primary_rt_date`, `salvage_rt_date`, `local_therapy`, `first_note_adt_date`, `n_psa_values`, `review_status`
  - each date column has its `_precision` and `_source` companions
  - dates use `tasks/onset.earliest`

### 3. Statement gate: new `tasks/disease_course/statement_gate.py` (versioned)

- Reuses `tasks/nepc/ne_gate.screen_asserted` for sentence-bounded negation and hedge screening of BCR and progression statements. It adds item-specific rejects:
  - "until / time to / upon progression"
  - "progression-free"
  - "if progresses"
  - planning cues
- CRPC statements go through the unchanged `mcrpc_gate.screen_castration_resistant_quote`.
- A new mCRPC term regex rejects nmCRPC and accepts "progressed to / developed mCRPC".
- Imaging uses a new certainty-aware gate, not the unchanged NE assertion gate, which rejects "suspicious", "probable" and "likely". Accept definite findings and patient-specific probable findings, including suspicious PSMA findings. Continue to reject negation, equivocal/indeterminate findings, hypothetical or planned assessments, and non-patient statements. An LLM certainty enum cannot override contradictory quote text.
- `annotate_trial_context` attaches `_trial_context_reason`; it does not filter findings. A shared course-evidence filter explicitly excludes every row with a non-null reason before building course labels, merging note PSAs, constructing therapy episodes or evaluating any endpoint. Apply quote-level trial screening as well, and retain rejection reasons for audit.
- No existing gate is edited.

### 4. Builders (no LLM)

Each builder:
- reads its inputs through `load_label_inputs`
- applies the shared course-evidence filter to all item types, including numeric PSA and therapy facts
- writes `<x>_labels.parquet` with one row per cohort DFCI_MRN (Int64), ISO dates, `_precision`/`_source` columns and `review_status`
- sets labels to null when the review failed or did not run
- writes a `.meta.parquet` with its versions, the course `run_config`, the structured source fingerprint and all resolved rule defaults (clearance threshold, baseline window, confirmation/flare policy, exposure assumptions and registry conflict policy)

Shared PSA logic, in a new `tasks/disease_course/psa.py`:
- `merge_psa(labs, note_psas)`:
  - reconciles measurements within each patient by specimen/accession identity when available; otherwise day-precision date, value and comparator identify a duplicate representation
  - also treats a note value as a restatement of a LABS draw when the LABS value, rounded to the note's stated decimal precision, equals the note value (same comparator class) and the LABS draw falls within the 7 days before the note's resolved date (configurable, recorded in metadata). This catches visit-dated and rounded restatements ("PSA 0.2" on Mar 4 for a LABS 0.23 on Mar 1). When several LABS draws match, the closest preceding one wins. A note value matched this way can never act as a separate draw or as confirmation.
  - prefers LABS for a matched measurement and preserves both sources in its audit trail; an unmatched note value remains eligible even if LABS has another draw within 30 days
  - same-day discrepancies without enough identity to distinguish separate draws are retained as conflicts, not resolved by picking a favorable value; unresolved conflicts cannot establish a threshold crossing
  - excludes year-precision values from computed rules while retaining their source findings for audit
  - preserves month precision for audit; uncertain-date observations cannot establish separate draws, a required ordering or a minimum interval unless their date bounds prove it
  - repeated quotes for one draw do not count as confirmation; preserve comparator and provenance through deduplication
- Threshold comparisons respect censoring: `<0.2` does not meet PSA ≥0.2. Unknown detection limits cannot establish clearance or a nadir. For rise/nadir calculations, bounded values qualify only when the bounds establish the result; do not substitute their limits as exact values.
- Rule functions:
  - `persistent_psa`: the first evaluable PSA 30–180 days after RP is ≥0.1 without earlier observed clearance. A rise after clearance is not persistence. The window and threshold are project defaults; if no usable measurement exists in that window, persistence is unknown, not false. Preserve the historical persistence flag after subsequent clearance.
  - `psa_clearance`: an observed post-RP PSA demonstrably <0.1 (including a usable `<0.1` result), or a dated post-RP `undetectable` note finding with day or month precision whose date bounds fall after RP. An undetectable result still cannot set a numeric nadir. This threshold and the requirement for clearance are project defaults for distinguishing recurrence from persistence.
  - `aua_bcr`: apply the project's operational ≥0.2 initial and later ≥0.2 confirmation thresholds only to separate draws after documented clearance; date at the first qualifying rise. Without clearance, retain elevated PSA findings but leave the PSA-derived BCR basis unevaluable. A persistent course that later clears and recurs can qualify at the later recurrence, never at the original persistent elevation.
  - `phoenix_bcr`: PSA ≥ preceding post-RT nadir + 2 after primary RT, dated at the qualifying measurement; never use a future nadir.
  - `pcwg3_progression`: select the latest evaluable PSA before therapy start within a configurable 30-day baseline window. A same-day value is eligible only if its pretreatment timing is established; otherwise exclude it from baseline selection. Record baseline as unavailable if no eligible value remains. Do not substitute the first on-treatment PSA for a missing baseline.
    - After a decline from baseline, compare the candidate with the preceding on-treatment nadir. With no decline, compare with pretreatment baseline. Both branches require ≥25% and ≥2 ng/mL increases.
    - Retain all early measurements for decline/nadir determination. The project's conservative default permits a progression candidate only after 12 weeks and requires a second qualifying draw ≥3 weeks later in the same therapy episode for either branch. Compare confirmation against the candidate's reference threshold; an intervening value below that threshold invalidates that candidate. Date the event at the first eligible rise and retain its confirmation date. This common timing/confirmation policy is a project adaptation of PCWG3, not a claim that its two branches have identical requirements.
    - Missing baseline, ambiguous timing or unavailable confirmation leaves the derived basis unevaluable; independently gated clinician progression statements remain usable. Store baseline, nadir, candidate and confirmation identities, branch and rule version.

Add `both_hold(rows)` to `tasks/onset.py`:
- date = the latest component date
- precision = the coarsest precision among components that are not certainly earlier than it

**`tasks/bcr/` (`criteria.py`, `build_bcr_labels.py`)**
- Bases: AUA after RP, Phoenix after primary RT, statement, imaging.
- The earliest eligible basis wins. Output columns: `bcr_date`, `bcr_basis`, `bcr_bases`, `persistent_psa`, `psa_clearance_date`, `psa_bcr_status` (met/not_met/unevaluable), `local_therapy`, `index_date` (the RP or RT date), and audit columns for each basis and exclusion reason.
- A `bcr_statement` with `bcr_kind="persistent_psa"` only supports the persistence flag. A separately asserted recurrence or imaging-confirmed recurrence may supply its own basis without a captured PSA clearance measurement; an image of ongoing residual disease does not establish a new recurrence.
- PSA and imaging bases require the index date; a statement does not.

**`tasks/progression/` (`events.py`, `build_progression_events.py`)**
- `progression_events.parquet` has one row per event. Sources:
  - gated statements
  - gated radiographic progression findings; do not promote every `imaging_recurrence` item into a progression event
  - PCWG3 PSA events per therapy episode; reset baseline/nadir at a new agent/regimen start or supported restart, not at every refill or repeated depot dose
  - death from the registry
- Shared exposure logic in `tasks/disease_course/therapy.py` reconciles structured medications with gated `therapy_start`, `therapy_current` and `therapy_stop` findings:
  - Use documented administration/formulation schedules, supply or stop dates where available. A single monthly dose must not become six months of established exposure. Separate oral and injectable ADT.
  - Repeated administration records of the same depot agent establish supported coverage from the observed inter-dose interval: each dose is covered until the next one, and the last dose of a run is covered for that patient's median observed interval (capped by the longest standard formulation, 6 months). Gaps longer than about 1.5× the observed interval end supported coverage and leave the remainder estimated. A single isolated record stays estimated unless formulation or schedule fields establish its duration.
  - A current-therapy statement establishes use at its dated assessment; it does not authorize indefinite coverage. An explicit stop ends reported use; any remaining documented depot coverage is separately identified. Unresolved conflicting evidence stays uncertain.
  - Without sufficient details, record an estimated interval and its assumptions only. A gap over 270 days may split an estimated ADT episode, but shorter gaps do not prove continuous exposure. No blanket 180-day ADT or 90-day other-drug window establishes active treatment.
- Each event carries `therapies_active`, `therapies_possible`, nullable `adt_active`, `exposure_basis` and `exposure_certainty`. Unknown coverage is not false, and estimated coverage alone cannot support PSA-derived CRPC.
- `progression_labels.parquet`, per patient: `review_status`, `death_date`, `last_alive_date`, registry conflict flags, and the first date of each event type.
- Helper `first_event_after(events, index_dates, types)` for the downstream PFS build.

**`tasks/mcrpc/` (`criteria.py`, `build_mcrpc_labels.py`)**
- CRPC holds when either of these is true:
  - PCWG3 progression with supported active ADT at the rise and confirmation, rather than an estimated medication tail. A testosterone value ≥50 ng/dL within 60 days of the rise vetoes this derived basis. A value <50 in the same window supports a measured-castrate tier; if testosterone is absent from that window, an assumed-castrate tier is allowed only with independent active-ADT evidence from documented coverage or dated current-treatment statements. Record the testosterone dates and exposure evidence; estimated exposure alone leaves this basis unevaluable.
  - a gated CRPC statement.
- The metastatic date comes from `--met-source {avpc,met_dx}`.
- `mcrpc_date` is the earliest eligible primary basis, with basis and certainty recorded:
  - the both-hold date
  - a dated mCRPC statement
- A documented prostate-cancer treatment start from the versioned proxy allowlist (cabazitaxel, radium-223, sipuleucel-T) is a `drug_proxy` basis that acts as a cap:
  - with no primary basis, it supplies `mcrpc_date`
  - when it is certainly earlier than the primary date, it supplies `mcrpc_date` (basis `drug_proxy`) and sets `proxy_precedes_primary=true`; the primary date and basis stay in audit columns
  - otherwise the primary basis wins
  - Keep this inference distinct from measured/statement-supported disease status; record the agent, indication evidence and policy version.
- Lu-177-PSMA is excluded from the drug-only proxy: Pluvicto gained an FDA indication in hormone-sensitive metastatic disease on July 31, 2026. Its use may still be recorded as therapy; CRPC requires independent evidence.
- Metadata also records `met-source`, the selected upstream label artifact's identity/hash, and its label metadata and criteria versions. A label-only upstream rebuild can change the metastatic date without changing its extraction fingerprint.

### 5. AVPC label columns (additive)

- `tasks/avpc/criteria.py::evaluate_patient` gains `metastatic_date_precision`, `metastatic_date_source`, `crpc_date_precision` and `crpc_date_source`, all read from the row `_state` returns.
- Add all four fields as `pl.Utf8` in `tasks/avpc/run_avpc_extraction.py::LABEL_SCHEMA`. The shared serializer keeps only declared schema keys; evaluator changes alone would discard the new fields.
- Bump `CRITERIA_VERSION` and apply it with `--rebuild-labels-only`; no LLM calls are needed.
- Keep `FINDING_SCHEMA`, prompts and extraction fingerprint inputs unchanged. The label schema addition must not invalidate saved extraction runs.

### 6. Notebook and docs

- `notebooks/disease_course.ipynb`, cloned from `avpc.ipynb`. It runs the collector, the runner, then the three builders.
- Update CLAUDE.md and README with the commands and a short architecture paragraph.

## Files

**New**
- `preprocessing/structured.py`
- `preprocessing/cli/discover_structured_codes.py`
- `preprocessing/cli/collect_course_notes.py`
- `tasks/disease_course/{__init__,prompts,run_course_extraction,statement_gate,psa,therapy}.py`
- `tasks/bcr/{__init__,criteria,build_bcr_labels}.py`
- `tasks/progression/{__init__,events,build_progression_events}.py`
- `tasks/mcrpc/{__init__,criteria,build_mcrpc_labels}.py`
- `notebooks/disease_course.ipynb`
- tests

**Modified**
- `preprocessing/config.py`
- `preprocessing/triggers.py`
- `tasks/note_extraction.py`
- `tasks/onset.py`
- `tasks/avpc/criteria.py`
- `tasks/avpc/run_avpc_extraction.py` (label schema only)
- CLAUDE.md, README.md

## Reused utilities

| What | Where |
|---|---|
| Evidence collection | `preprocessing/evidence.py` (`collect_note_evidence`, `add_collector_args`, `note_task_paths`, `downstream_paths`) |
| Per-note runner | `tasks/note_extraction.py`: `TaskSpec`, grounding, ledger, resume, `annotate_trial_context`, `patient_review_status` |
| Dates | `tasks/onset.py` (`earliest`, `prefer_onset`, `date_interval`); `preprocessing/notes.to_iso_date` |
| Gates | `tasks/nepc/ne_gate.screen_asserted`; `tasks/longitudinal_subtype/mcrpc_gate.py`; `tasks/trial_guard.py` |
| Number check and planning cues | `tasks/avpc/criteria.py`: `number_in_quote`, `ADT_PLANNED` |
| Field validation pattern | `tasks/avpc/run_avpc_extraction.py` |

## Verification

1. Run `pytest`. New tests cover:
   - an end-to-end run with FakeProvider
   - nepc/avpc fingerprints unchanged with the default `max_findings_per_note`
   - the structured loaders on synthetic parquet: sentinels, comparator bounds, units, source identities and optional exposure fields
   - registry reduction is invariant to row order, keeps the latest last-alive date, retains conflicting death candidates and flags last-alive-after-death inconsistencies
   - a note PSA rise on March 1 and a separate LABS confirmation on March 25 both survive reconciliation; duplicate quotes of one draw cannot confirm an event; ambiguous same-day conflicts remain unevaluable
   - RP followed by PSA 0.3 at day 45 and 0.4 at day 80 sets persistence without PSA-derived BCR; later clearance followed by a new confirmed rise dates BCR at that later rise; missing clearance remains unevaluable
   - AUA/Phoenix thresholds, censored values, uncertain dates and preceding-nadir selection
   - PCWG3 baseline 10 with no decline and values 15 at week 13 and 16 at week 17 yields progression at week 13; baseline 50, week-4 nadir 1, week-13 PSA 4 and week-17 PSA 5 retains the early nadir and dates progression at week 13; missing baseline, early rises and confirmation outside the episode do not create derived events
   - one monthly ADT record in January does not establish active ADT for May/June progression; stop/current statements, oral ADT, unknown formulation and absent testosterone preserve the intended certainty tiers
   - Lu-177-PSMA alone cannot establish mCRPC; independent CRPC evidence can; an eligible drug proxy supplies the date when no primary basis exists, and caps it with `proxy_precedes_primary=true` when it is certainly earlier (cabazitaxel 2019 + rules 2021 → 2019)
   - "PSA undetectable" 2 months after RP (no number) counts as clearance; a later confirmed rise ≥0.2 then yields AUA BCR; an undetectable result never sets a numeric nadir
   - a note "PSA 0.2" dated Mar 4 matches a LABS 0.23 on Mar 1 and cannot confirm it; a note "PSA 0.4" on Mar 25 with no matching LABS draw is a separate draw
   - depot ADT administrations every ~90 days give supported coverage through the last dose + 90 days; a 400-day gap ends supported coverage
   - suspicious/probable PSMA recurrence survives the imaging gate; negated, equivocal and hypothetical findings do not; stable repeat imaging is not a new progression event
   - source-context trial rejections exclude short cropped PSA and therapy quotes as well as statements from all course/endpoint builders, including label-only rebuilds
   - the BCR, progression and mCRPC builders, their audit fields, and upstream-label provenance
   - Rust-regex compatibility of `COURSE_TRIGGER_REGEX` through `combined_text_pattern` and polars
   - all four new AVPC precision/source columns survive a label-only rebuild and Parquet round trip; existing label values and extraction fingerprints remain unchanged
2. On the cluster, run `discover_structured_codes.py` to confirm the PSA, testosterone and medication codes.
3. Pilot run: `collect_course_notes.py --mrns …`, then `run_course_extraction.py --mrns … --provider vertex_ai`, then the three builders. Spot-check dates against the charts.
4. Run `run_avpc_extraction.py --rebuild-labels-only` and confirm the new columns.

## Rule references

- [PCWG3, Table 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC4872347/#T5): distinguishes post-decline nadir from the no-decline baseline. The common 12-week onset floor, confirmation requirement for both branches and 30-day baseline window above are explicit project defaults.
- [FDA Lupron Depot prescribing information](https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/020517s045lbl.pdf): formulations have different administration schedules, so a drug name alone cannot justify six months of exposure.
- [FDA Pluvicto indication expansion, July 31, 2026](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-lutetium-lu-177-vipivotide-tetraxetan-androgen-receptor-pathway-inhibitor-therapy): metastatic hormone-sensitive use prevents Lu-177-PSMA from serving as an unconditional mCRPC proxy.

## Out of scope (follow-up)

- Registering the BCR, PFS and mCRPC endpoints in `survival_common` (the downstream checklist of about 10 steps).
- Building PFS as progression OR death from `progression_events` against the chosen index date.
- Rebasing BCR to the RP/RT index date, following `build_somatic_gleason_inputs.py`.
