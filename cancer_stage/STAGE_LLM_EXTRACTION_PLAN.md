# Cancer Stage LLM Extraction Plan

## Goal

Extend `cancer_stage/extract_stage_notes.py` so it:

1. Scans clinical notes for all mentions of cancer stage.
2. Extracts compact context around each stage mention.
3. Attaches note date, note type, and provenance to each context.
4. Groups contexts per patient into payload-sized chunks.
5. Calls the LLM once per chunk to produce a stage timeline.

The intended output is a structured, provenance-preserving cancer-stage extraction rather than only a table of notes containing stage terms.

## Shared Infrastructure to Reuse

The `shared/` directory already provides the plumbing the original plan proposed to build. Use it directly.

### `shared/llm_helpers.py`

| Symbol | Purpose |
| --- | --- |
| `load_notes()` | Unified note loader (bundle → CSV → raw OncDRS JSON). Handles all CLINICAL_TEXTS_* path variants. |
| `discover_raw_text_files()` | Discovers raw JSON files and infers `NOTE_TYPE` from filename. |
| `build_raw_note_row()` | Extracts `NOTE_TYPE`, `CLINICAL_TEXT`, `RAW_NOTE_ID`, `RPT_TYPE`, and all standard columns from a raw doc dict. |
| `build_client()` | Azure OpenAI client with `DefaultAzureCredential`. |
| `call_with_retry()` | Retry wrapper with rate-limit backoff and content-filter handling. |
| `parse_json_response()` | JSON extraction with regex fallback. |
| `CLINICAL_SAFETY_CONTEXT` | Standard IRB-approved safety preamble to prepend to system prompts. |
| `to_iso_date()` | Normalizes any date value to ISO string or `None`. |
| `normalize_mrn_column()` | Coerces and drops non-numeric MRN rows. |
| `load_selected_mrns()` | Loads MRN filter from CLI arg or file. |
| `NOTE_BUNDLE_COLUMNS` | Canonical column set for all note-loading outputs. |
| `SNIPPET_CONTEXT_CHARS` | Default context window: 2000 chars on each side of a trigger match. |

### `shared/longitudinal_helpers.py`

| Symbol | Purpose |
| --- | --- |
| `find_matches(text, trigger_regex)` | Returns sorted `(label, start, end)` tuples for all hits of a `{label: pattern}` dict. |
| `iter_note_snippets(notes_df, trigger_regex)` | Yields one snippet dict per trigger-bearing note. Applies `clean_note(text, note_type=note_type)` (fixing the missing `note_type` bug in the current script). Deduplicates copy-forward notes per patient by exact snippet text, keeping the earliest `note_date`. |
| `group_patient_snippets(notes_df, trigger_regex)` | Groups deduped snippets by patient into payload-sized chunks (chronological, greedy packing to `DEFAULT_PAYLOAD_MAX_CHARS=60000`). No snippet is ever silently dropped. |
| `resolve_date(stated_date, note_date)` | Returns `(iso_date, date_source)` where `date_source` is `"stated"` or `"note_date"`. |
| `flatten_ws(value)` | Collapses whitespace for TSV-safe storage of verbatim quotes. |

## Stage Trigger Regex

Define `STAGE_TRIGGER_REGEX` in `extract_stage_notes.py` in the same `{label: pattern}` shape as `TRIGGER_REGEX` in `llm_helpers.py`. Pass it wherever `trigger_regex` is expected.

Suggested categories:

```python
STAGE_TRIGGER_REGEX = {
    "stage_group": (
        r"\b(?:clinical|pathologic|pathological|c|p|yp|yc)?\s*"
        r"stage\s+(?:IV[ABC]?|III[ABC]?|II[ABC]?|I[ABC]?|[0-4][ABC]?)"
        r"|\bstage\s+(?:one|two|three|four)\b"
    ),
    "staging_system": (
        r"\b(?:AJCC|FIGO|Ann\s+Arbor)\s+stage"
    ),
    "tnm": (
        # Require at least T+N or T+M to avoid bare "T2" imaging descriptors.
        r"\b[cpyry]{0,2}T[0-4][a-z]?\s*N[0-3X]\s*M[01X]"
        r"|\b[cpyry]{0,2}T[0-4][a-z]?\s*N[0-3X]\b"
        r"|\bM1[abc]?\b"
    ),
    "limited_extensive": (
        r"\b(?:limited|extensive)\s+stage\b"
    ),
}
```

False positives in the `tnm` and `limited_extensive` categories are acceptable; the LLM will distinguish true staging records from incidental mentions. The `trigger_category` field in each snippet tells the model which type of evidence to weight.

**Rationale for including TNM-only:** Pathology reports commonly encode staging as `pT3bN0M0` with no surrounding "stage" text, and `M1` alone implies stage IV. Missing these would cause systematic under-recall on pathology notes. Requiring at least two TNM components (`TxNx`, `TxMx`) avoids the most common false-positive class ("T2-weighted MRI", "T2 vertebral body").

## Proposed Data Flow

```text
load_notes() or raw OncDRS scan
  -> iter_note_snippets(notes_df, STAGE_TRIGGER_REGEX)
     [clean_note(text, note_type), find_matches, build_snippet, dedup by (mrn, snippet)]
  -> write raw evidence table (before any LLM calls)
  -> group_patient_snippets(notes_df, STAGE_TRIGGER_REGEX)
     [chronological sort, greedy chunk to DEFAULT_PAYLOAD_MAX_CHARS]
  -> one LLM call per chunk
  -> write structured stage output
```

## Raw Evidence Table

Write this before any LLM calls. It is the audit trail for the scanning stage and allows the LLM layer to be re-run independently.

Columns derived from snippet dicts yielded by `iter_note_snippets`:

```text
note_uid          (from _note_uid — stable per-note identifier)
DFCI_MRN
note_date         (ISO, from EVENT_DATE)
note_type         (NOTE_TYPE — Clinician / Imaging / Pathology / Unknown)
trigger_categories (list; one or more of stage_group, staging_system, tnm, limited_extensive)
snippet           (context window around all matches in the note)
```

## LLM Payload Shape

Match the shape used by the existing longitudinal pipelines. Each chunk passed to the model is a list of snippet dicts:

```json
{
  "patient_mrn": 123,
  "stage_contexts": [
    {
      "note_date": "2021-04-15",
      "note_type": "Pathology",
      "trigger_categories": ["stage_group", "tnm"],
      "snippet": "..."
    },
    {
      "note_date": "2022-01-10",
      "note_type": "Clinician",
      "trigger_categories": ["tnm"],
      "snippet": "..."
    }
  ]
}
```

## LLM Task

Ask the model to extract a stage timeline. A deterministic post-processing step derives patient-level summaries (latest stage, highest-confidence stage).

Suggested output fields per finding:

```text
cancer_type
stage_group         (I / II / III / IV or null)
tnm                 (e.g. "cT3N1M1" or null)
stage_date          (ISO or null)
date_source         (resolve_date return value: "stated" | "note_date" | "unknown")
is_historical_reference  (true if the note is recounting a prior staging, not current)
supporting_quote    (verbatim)
confidence          (high | medium | low)
rationale
```

Example model output:

```json
{
  "stage_findings": [
    {
      "cancer_type": "prostate cancer",
      "stage_group": "IV",
      "tnm": "cT3N1M1",
      "stage_date": "2021-04-15",
      "date_source": "note_date",
      "is_historical_reference": false,
      "supporting_quote": "...",
      "confidence": "high",
      "rationale": "The context explicitly documents stage IV disease."
    }
  ]
}
```

## File Layout

```text
cancer_stage/
  extract_stage_notes.py     # define STAGE_TRIGGER_REGEX, scan notes, write evidence table
  run_stage_extraction.py    # call LLM on patient chunks, write structured stage output
  prompts.py                 # STAGE_SYSTEM_PROMPT and output schema
shared/
  llm_helpers.py             # LLM client, note loading, snippet building (reused)
  longitudinal_helpers.py    # iter_note_snippets, group_patient_snippets (reused)
  utils.py                   # clean_note (reused)
```

`extract_stage_notes.py` defines `STAGE_TRIGGER_REGEX` and calls `iter_note_snippets` / `group_patient_snippets` directly. It does not reimplement file scanning, context windowing, deduplication, or chunking.

`run_stage_extraction.py` imports `build_client`, `call_with_retry`, `parse_json_response`, and `CLINICAL_SAFETY_CONTEXT` from `shared/llm_helpers.py`. It iterates over patient chunks, builds the JSON payload, calls the model, and writes the structured output.

## Open Decisions

1. **Stage timeline vs. single best stage**: build the timeline extractor first; derive patient-level summaries deterministically in post-processing. *(Decided.)*

2. **Cancer-agnostic vs. cancer-type-tuned**: the regex scan is cancer-agnostic. The LLM prompt should ask for `cancer_type` in each finding so post-processing can filter by primary site without re-running the model.

3. **TNM-only mentions**: include them as a distinct `trigger_category`. Requiring two TNM components in the pattern avoids the most common false-positive class. *(Decided.)*

4. **Chunking for very long charts**: handled by `group_patient_snippets`. Chunk-level outputs for the same patient need a final deduplication pass on `(cancer_type, stage_group, tnm, stage_date)` before being written as the patient-level timeline.
