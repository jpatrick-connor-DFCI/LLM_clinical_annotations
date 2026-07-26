# Binary NEPC Classifier

Classifies each prostate cancer patient into one of four buckets with a single LLM call:

- **nepc** — neuroendocrine / small-cell prostate cancer or documented histologic transformation
- **avpc** — aggressive-variant / anaplastic language or one or more Aparicio C1–C7 features
- **biomarker** — platinum-triggering somatic biomarker: BRCA1, BRCA2, or PALB2
- **conventional** — none of the above

Precedence: `nepc > avpc > biomarker > conventional` (the LLM applies it in the same call).

Independently, each patient is also flagged for two separate annotations that can co-occur with any primary label:

- `has_non_prostate_primary` — synchronous/metachronous non-prostate primary (e.g., NSCLC, colorectal, urothelial, RCC, lymphoma).
- `has_molecular_avpc` — ≥ 2 somatic alterations among {PTEN, TP53, RB1}. Fully independent of `has_avpc` — does not change `primary_label` or `avpc_criteria`.

AVPC C-criteria refinements:

- **C2** (visceral pattern) requires lung / adrenal / brain / pleural / peritoneal metastasis; liver-only involvement does NOT qualify. When C2 is set, `visceral_met_pattern` records either `visceral_only` or `visceral_and_bone`.
- **C4** (bulky disease) is restricted to bulky lymphadenopathy / nodal disease OR a prostate / pelvic mass with a documented measurement ≥ 5 cm.

## Files

```text
binary_NEPC/
  compile_patient_snippets.py      # required stage 1: save ranked trigger snippets
  snippet_bundle.py                # versioned snippet artifact I/O
  run_NEPC_classifier.py           # stage 2: classify the saved snippets
  compile_prostate_note_bundle.py  # optional: pre-compile raw OncDRS notes into a gzip bundle
shared/
  llm_helpers.py                   # config, triggers, prompt, snippet builder, LLM client
  utils.py                         # shared note cleaning
```

## How it works

1. `compile_patient_snippets.py` loads the selected notes, trigger-scans them,
   ranks the matches per patient, and saves
   `LLM_NEPC_classifier_patient_snippets.json.gz`.
2. The saved artifact contains both the triggered snippets and the complete
   cohort MRN list, including patients with no triggers.
3. `run_NEPC_classifier.py` reads only that artifact. It sends triggered
   patients to Gemini and labels no-trigger patients as `conventional`.

The classifier does not load, clean, or scan source notes. Rebuild the snippet
artifact explicitly whenever the cohort, notes, triggers, or snippet settings
change.

## Recommended run

```bash
# Stage 1: compile and save patient snippets
python binary_NEPC/compile_patient_snippets.py \
  --mrn-file path/to/prostate_mrns.txt

# Stage 2: classify only from the saved snippets
python binary_NEPC/run_NEPC_classifier.py \
  --mrn-file path/to/prostate_mrns.txt \
  --max-workers 4
```

Use `--output-path` in stage 1 and the matching `--snippets-path` in stage 2
when the snippet artifact lives outside the default output directory. The
optional raw-note bundle remains an input optimization for stage 1 only.

## Outputs

By default, `binary_NEPC` writes to `/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_NEPC_labels/`.
Set `BINARY_NEPC_OUTPUT_DIR` to override it. The legacy
`CAIA_COMPASS_NEPC_CLASSIFIER_OUTPUT_DIR` is also accepted.

- `LLM_NEPC_classifier_note_bundle.json.gz` — optional pre-compiled note bundle
- `LLM_NEPC_classifier_patient_snippets.json.gz` — required saved snippet artifact
- `LLM_NEPC_classifier_labels.tsv` — one row per patient with the final classification, supporting quotes, confidence, and rationale
- `LLM_NEPC_classifier_failed_patients.tsv` — current unlabeled patients whose LLM call errored; successful retries are removed

`LLM_NEPC_classifier_labels.tsv` columns:

```text
DFCI_MRN, primary_label,
has_nepc, has_avpc, has_biomarker, has_molecular_avpc, has_non_prostate_primary,
biomarker_genes, avpc_criteria, visceral_met_pattern, non_prostate_primary_types,
supporting_quotes, supporting_quote_dates,
confidence, rationale, num_snippets
```

The pipeline is resumable: re-running skips MRNs already present in `LLM_NEPC_classifier_labels.tsv`. Failed patients are retried by a normal resumed run, or use `--retry-failures` to run only patients in the failed-patients TSV. Successful retries are appended to labels and removed from the failed/unlabeled TSV. Use `--overwrite` to start fresh.

## Useful flags

```text
compile_patient_snippets.py:
--max-notes-per-patient N    # cap selected snippets per patient (default 75)
--output-path PATH           # saved snippet artifact
--overwrite                  # explicitly rebuild an existing artifact

run_NEPC_classifier.py:
--snippets-path PATH         # required precompiled snippet artifact
--max-workers N              # concurrent patient classifications (default 4)
--limit-mrns N               # cap how many MRNs to process this run
--model NAME                 # override the Gemini model
--retry-failures             # only rerun patients in the failed-patients TSV
--overwrite                  # delete prior labels/failures and start over
```

## Notes

- All triggers are matched on `clean_note`-cleaned text during snippet
  compilation. The classifier preserves and reports the saved snippet count.
- No structured labs, genomics tables, medication tables, or PSA tables are used in this classifier — all signal comes from note text.
- `cisplatin` and `carboplatin` are no longer used as triggers; biomarker selection is driven by the molecular terms above.
