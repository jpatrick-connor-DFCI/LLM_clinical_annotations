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
  run_NEPC_classifier.py           # main entrypoint
  compile_prostate_note_bundle.py  # optional: pre-compile raw OncDRS notes into a gzip bundle
shared/
  llm_helpers.py                   # config, triggers, prompt, snippet builder, LLM client
  utils.py                         # shared note cleaning
```

## How it works

1. Load notes for the requested MRNs (from a pre-compiled gzip bundle if present, otherwise raw OncDRS JSONs).
2. Per note, scan for any of the combined NEPC + AVPC + biomarker trigger regexes and build a snippet window around the matches.
3. Per patient, rank triggered notes by `(# trigger categories, # triggers, recency)` and keep the top N (default 30).
4. Send all selected snippets to the LLM in a single call. The model returns the patient's primary label, per-category booleans, supporting quotes, and a rationale.
5. Patients with no triggered notes are written out as `conventional` without an LLM call.

## Recommended run

```bash
# (one-time) compile a gzip note bundle so re-runs don't re-scan raw OncDRS JSON
python binary_NEPC/compile_prostate_note_bundle.py --mrn-file path/to/prostate_mrns.txt

# classify
python binary_NEPC/run_NEPC_classifier.py --mrn-file path/to/prostate_mrns.txt --max-workers 4
```

If the bundle lives elsewhere: `--note-bundle-path path/to/LLM_NEPC_classifier_note_bundle.json.gz`.

## One-command raw run

```bash
python binary_NEPC/run_NEPC_classifier.py --mrn-file path/to/mrns.txt --max-workers 4
```

When no bundle exists at the expected path, the pipeline falls back to scanning raw OncDRS JSONs directly.

## Outputs

By default, `binary_NEPC` writes to `/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_NEPC_labels/`.
Set `BINARY_NEPC_OUTPUT_DIR` to override it. The legacy
`CAIA_COMPASS_NEPC_CLASSIFIER_OUTPUT_DIR` is also accepted.

- `LLM_NEPC_classifier_note_bundle.json.gz` — optional pre-compiled note bundle
- `LLM_NEPC_classifier_labels.tsv` — one row per patient with the final classification, supporting quotes, confidence, and rationale
- `LLM_NEPC_classifier_failed_patients.tsv` — appended for any patient whose LLM call errored

`LLM_NEPC_classifier_labels.tsv` columns:

```text
DFCI_MRN, primary_label,
has_nepc, has_avpc, has_biomarker, has_molecular_avpc, has_non_prostate_primary,
biomarker_genes, avpc_criteria, visceral_met_pattern, non_prostate_primary_types,
supporting_quotes, supporting_quote_dates,
confidence, rationale, num_snippets
```

The pipeline is resumable: re-running skips MRNs already present in `LLM_NEPC_classifier_labels.tsv`. Use `--overwrite` to start fresh.

## Useful flags

```text
--max-notes-per-patient N    # cap selected snippets per patient (default 30)
--max-workers N              # concurrent patient classifications (default 4)
--limit-mrns N               # cap how many MRNs to process this run
--model NAME                 # override the Azure OpenAI deployment (default gpt-4o)
--overwrite                  # delete prior labels/failures and start over
```

## Notes

- All triggers are matched on `clean_note`-cleaned text. Each note's snippet is capped at ~30000 chars; per-patient snippet counts are capped (default 30), so a single LLM call typically sees a larger focused context while still staying under the patient payload cap.
- No structured labs, genomics tables, medication tables, or PSA tables are used in this classifier — all signal comes from note text.
- `cisplatin` and `carboplatin` are no longer used as triggers; biomarker selection is driven by the molecular terms above.
