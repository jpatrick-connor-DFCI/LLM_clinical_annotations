# Preprocessing Extraction Plan

Factor all non-API preprocessing out of `dfci_gpt/` and `vertex_ai/` into a
standalone `preprocessing/` package, reduce the provider trees to thin API
adapters, and drive every task from a notebook with a provider toggle.

**Conflict rule for this refactor: where the two trees disagree, the dfci_gpt
implementation wins.** It is a strict superset. The single exception is snippet
sizing, which is not a correctness conflict but a per-pipeline tuning choice —
see [Snippet sizing](#snippet-sizing).

---

## 1. Audit findings

### 1.1 The duplication is almost entirely accidental

All 16 parallel files compared:

| File | dfci_gpt | vertex_ai | Status |
|---|---:|---:|---|
| `shared/utils.py` | 193 | 193 | identical |
| `shared/compile_prostate_notes.py` | 122 | 122 | identical |
| `shared/compile_text_for_LLM_review.py` | 45 | 45 | identical |
| `shared/longitudinal_helpers.py` | 224 | 224 | identical |
| `shared/__init__.py` | 0 | 0 | identical |
| `binary_NEPC/snippet_bundle.py` | 121 | 121 | identical |
| `binary_NEPC/compile_prostate_note_bundle.py` | 90 | 90 | identical |
| `cancer_stage/extract_stage_notes.py` | 391 | 391 | identical |
| `cancer_stage/prompts.py` | 60 | 60 | identical |
| `cancer_stage/run_stage_extraction.py` | 421 | 421 | identical |
| `gleason_score/extract_gleason_timeline.py` | 414 | 414 | identical |
| `longitudinal_NEPC/extract_avpc_nepc_timeline.py` | 440 | 445 | docstring only |
| `binary_NEPC/compile_patient_snippets.py` | 114 | 107 | gpt ahead |
| `binary_NEPC/run_NEPC_classifier.py` | 318 | 309 | gpt ahead |
| `shared/llm_helpers.py` | 1000 | 793 | **the real fork** |
| `requirements.txt` | 7 | 5 | provider SDKs |

Eleven files are byte-identical. One differs only by a docstring. The entire
meaningful divergence lives in `llm_helpers.py`.

### 1.2 Every prompt and config constant is already identical

Verified by AST comparison of the assigned expressions:

| Constant | Result |
|---|---|
| `CLASSIFY_SYSTEM_PROMPT` | identical |
| `CLINICAL_SAFETY_CONTEXT` | identical |
| `TRIGGER_REGEX` | identical |
| `NOTE_BUNDLE_COLUMNS` | identical |
| `DEFAULT_DATA_PATH` | identical |
| `DEFAULT_OUTPUT_DIR` | identical |
| `DEFAULT_RAW_TEXT_PATHS` | identical |

No prompt text is at risk in this refactor. Nothing that shapes model output
differs between the trees.

### 1.3 Symbol-level diff of `llm_helpers.py`

```
only in vertex_ai: VERTEX_PROJECT, VERTEX_LOCATION
only in dfci_gpt:  DEFAULT_AZURE_OPENAI_ENDPOINT, DEFAULT_AZURE_OPENAI_API_VERSION,
                   resolve_note_source, scan_note_candidates, rank_patient_candidates,
                   _scan_note_row, _scan_note_chunk, _chunked, _snippet_cache_key,
                   _retry_after_seconds, _backoff_sleep
```

Apart from the two provider config constants, dfci_gpt is a **strict superset**.
Vertex has no unique logic to preserve. This is what makes "gpt wins" safe as a
blanket rule rather than a per-case judgment.

### 1.4 The provider seam is already narrow

`dfci_gpt/shared/llm_helpers.py` has an explicit `# LLM client` marker at line
907. Above it: paths, triggers, prompts, MRN parsing, note loading, snippet
building — all provider-neutral. Below it: three functions.

| Function | dfci_gpt | vertex_ai |
|---|---|---|
| `build_client()` | Azure AD token provider → `AzureOpenAI` | `genai.Client(vertexai=True, ...)` (new `google-genai` SDK) |
| `call_with_retry()` | `chat.completions.create` | `client.models.generate_content` (status-code-based retry) |
| `parse_json_response()` | — | **byte-identical, not provider-specific** |

Note: as of this plan, `vertex_ai/shared/llm_helpers.py` is mid-migration in the
working tree from the legacy `vertexai`/`GenerativeModel` SDK to the new
`google.genai` SDK (`genai.Client`, `genai_errors.APIError`, HTTP status codes
429/408/504 instead of typed `gcp_exceptions`). `providers/vertex_ai.py` is
built from the **current working-tree code**, not the older `vertexai.init()`
version this plan originally inspected.

Every consumer imports exactly `build_client`, `call_with_retry`,
`parse_json_response`, `DEFAULT_MODEL_NAME`; passes an opaque `client` plus
OpenAI-style `messages`; and receives `(text, error)` where errors are returned,
never raised. Vertex already adapts `messages` internally and returns `None` as
its client. **An adapter interface exists in practice — it just isn't named.**

### 1.5 Existing section markers map directly onto target modules

`dfci_gpt/shared/llm_helpers.py` is already sectioned by comment:

| Line | Marker | Destination |
|---:|---|---|
| 39 | `# Paths` | `preprocessing/config.py` |
| 80 | `# Azure OpenAI` | `providers/dfci_gpt.py` |
| 91 | `# Triggers` | `preprocessing/triggers.py` |
| 169 | `CLASSIFY_SYSTEM_PROMPT` | `tasks/binary_NEPC/prompts.py` |
| 310 | `# MRN parsing` | `preprocessing/notes.py` |
| 356 | `# Note text utilities` | `preprocessing/notes.py` |
| 405 | `# Raw / bundle loaders` | `preprocessing/notes.py` |
| 655 | `# Snippet building` | `preprocessing/snippets.py` |
| 907 | `# LLM client` | `providers/` |

The split follows boundaries the file already declares.

---

## 2. Target layout

```text
LLM_clinical_annotations/
  preprocessing/                   # NEW — zero API imports, enforced by test
    __init__.py
    config.py                      # paths, snippet-size profiles
    utils.py                       # clean_note
    notes.py                       # MRN parsing, load/clean/standardize/bundle
    triggers.py                    # TRIGGER_REGEX, find_trigger_matches, merge_windows
    snippets.py                    # build_snippet, scan, rank, cache
    longitudinal.py                # iter_note_snippets, group_patient_snippets, resolve_date
    bundles/
      __init__.py
      note_bundle.py
      snippet_bundle.py
    cli/
      compile_prostate_notes.py
      compile_prostate_note_bundle.py
      compile_patient_snippets.py
      extract_stage_notes.py
      collect_gleason_notes.py     # NEW — split out
      collect_nepc_notes.py        # NEW — split out

  providers/                       # NEW — only place API SDKs are imported
    __init__.py                    # get_provider(name)
    base.py                        # Provider protocol
    dfci_gpt.py                    # Azure OpenAI adapter
    vertex_ai.py                   # Vertex AI adapter
    response.py                    # parse_json_response (shared)

  tasks/                           # provider-agnostic runners + prompts
    binary_NEPC/{__init__,prompts,run_NEPC_classifier}.py
    cancer_stage/{__init__,prompts,run_stage_extraction}.py
    gleason_score/{__init__,prompts,build_gleason_timeline}.py
    longitudinal_NEPC/{__init__,prompts,build_nepc_timeline}.py

  notebooks/
    binary_NEPC.ipynb
    cancer_stage.ipynb
    gleason_score.ipynb
    longitudinal_NEPC.ipynb

  tests/
    test_no_api_imports.py
    test_snippet_pipeline.py
    test_symbol_coverage.py

  pyproject.toml
  requirements.txt                 # core: polars, python-dateutil, tqdm, ijson
  requirements-dfci_gpt.txt        # openai, azure-identity
  requirements-vertex_ai.txt       # google-cloud-aiplatform>=1.60
```

`dfci_gpt/` and `vertex_ai/` are deleted in the final step.

### 2.1 Where each existing file goes

| Current (dfci_gpt) | Destination |
|---|---|
| `shared/llm_helpers.py` L1–906 | `preprocessing/{config,triggers,notes,snippets}.py` |
| `shared/llm_helpers.py` L907–990 | `providers/dfci_gpt.py` |
| `shared/llm_helpers.py` `parse_json_response` | `providers/response.py` |
| `shared/llm_helpers.py` `CLASSIFY_SYSTEM_PROMPT`, `CLINICAL_SAFETY_CONTEXT` | `tasks/binary_NEPC/prompts.py`, `preprocessing/config.py` |
| `shared/utils.py` | `preprocessing/utils.py` |
| `shared/longitudinal_helpers.py` | `preprocessing/longitudinal.py` (minus API re-exports) |
| `shared/compile_prostate_notes.py` | `preprocessing/cli/compile_prostate_notes.py` |
| `shared/compile_text_for_LLM_review.py` | `preprocessing/cli/` |
| `binary_NEPC/snippet_bundle.py` | `preprocessing/bundles/snippet_bundle.py` |
| `binary_NEPC/compile_prostate_note_bundle.py` | `preprocessing/cli/` |
| `binary_NEPC/compile_patient_snippets.py` | `preprocessing/cli/` |
| `binary_NEPC/run_NEPC_classifier.py` | `tasks/binary_NEPC/` |
| `cancer_stage/extract_stage_notes.py` | `preprocessing/cli/` |
| `cancer_stage/prompts.py` | `tasks/cancer_stage/prompts.py` |
| `cancer_stage/run_stage_extraction.py` | `tasks/cancer_stage/` |
| `gleason_score/extract_gleason_timeline.py` | **split** — see §5 |
| `longitudinal_NEPC/extract_avpc_nepc_timeline.py` | **split** — see §5 |
| `vertex_ai/shared/llm_helpers.py` L738–782 | `providers/vertex_ai.py` |

Everything else in `vertex_ai/` is discarded as duplicate.

---

## 3. Conflict resolution — gpt wins

Applied wherever the trees disagree. Each item is a **behavior change for Vertex
runs**, all of them improvements Vertex simply never received.

### 3.1 From `llm_helpers.py`

- **`resolve_note_source()`** — bundle/csv/raw resolution as a reusable
  function. Vertex inlines this branching in `compile_patient_snippets.py`;
  the inline version is deleted.
- **`scan_note_candidates(..., max_workers=)`** — `ProcessPoolExecutor` parallel
  clean + trigger scan, with `_scan_note_row`, `_scan_note_chunk`, `_chunked`.
  Vertex scans serially inside `build_patient_snippets`.
- **`rank_patient_candidates()`** — ranking split out of `build_patient_snippets`
  as its own function.
- **`_snippet_cache_key()`** — gzip snippet cache, written to `.tmp` then
  atomically `replace()`d so a half-written cache is never left behind.

### 3.2 From `run_NEPC_classifier.py`

- **No-signal rows written after LLM submission.** dfci_gpt submits all LLM work
  first, then writes auto-conventional rows while calls are in flight. Vertex
  writes them before submitting, delaying time-to-first-call. Keep gpt's order,
  including its explanatory comment.
- **Stale-state repair.** dfci_gpt removes patients from the failure/unlabeled
  list when a retry succeeds; older runs left them stranded. Vertex lacks this.
- **Latest-error-only on repeated retries** — dfci_gpt keeps only the most
  recent error per patient rather than accumulating.

### 3.3 From `compile_patient_snippets.py`

- **`--scan-workers`** CLI flag (default `None` = all cores), threaded through to
  `scan_note_candidates` and recorded in bundle metadata.

### 3.4 Retry/backoff stays provider-specific

dfci_gpt's `_retry_after_seconds()` / `_backoff_sleep()` move into
`providers/dfci_gpt.py`, **not** into shared code. They parse the Azure
`Retry-After` header and error-body text; that is Azure-specific. Vertex keeps
its `gcp_exceptions.ResourceExhausted` / `DeadlineExceeded` handling.

This is the one place "gpt wins" does **not** mean "gpt's code replaces
vertex's" — the two adapters legitimately differ, which is exactly why they are
separate modules. Sharing here would mean parsing Azure headers off Google
exceptions.

### 3.5 Snippet sizing

The sizing constants live in the common `preprocessing/config.py` like
everything else in this section. What cannot collapse to a single shared
*value* is the number itself — and the reason is **per-task, not per-provider**.

#### The values are already task-scoped inside dfci_gpt alone

| Constant | `llm_helpers.py` (binary_NEPC) | `longitudinal_helpers.py` (stage/gleason/longitudinal) |
|---|---:|---:|
| `SNIPPET_CONTEXT_CHARS` | 750 | 6000 |
| `SNIPPET_MAX_CHARS` | 300000 | 30000 |
| `PATIENT_PAYLOAD_MAX_CHARS` | 300000 | 60000 (`DEFAULT_PAYLOAD_MAX_CHARS`) |

`shared/longitudinal_helpers.py:34-38` redefines these, and the redefinition is
live: `iter_note_snippets` (L87-88) and `group_patient_snippets` (L165-167) bind
them as keyword defaults and pass them into the **same** `build_snippet`
imported from `llm_helpers`. One shared function is already called with two
different sizings depending on the calling task — in dfci_gpt and vertex_ai
alike.

Both settings are deliberate. binary_NEPC makes one call per patient and wants
tight 750-char windows around each trigger with effectively no per-note cap. The
longitudinal tasks need wide 6000-char context to date events, but must cap each
note at 30k and chunk at 60k because they pack many notes into one call.
Collapsing them to one number degrades one task or the other.

**So "gpt wins" does not decide this axis** — the rule resolves gpt-vs-vertex
disagreements, and here gpt itself holds both values.

#### The actual gpt-vs-vertex conflict on this axis

| Constant | dfci_gpt | vertex_ai | Resolution |
|---|---:|---:|---|
| `merge_windows(gap_chars=)` | 300 | 80 | gpt wins → 300 |
| binary_NEPC context/max | 750 / 300k | 6000 / 30k | gpt wins → 750 / 300k |
| longitudinal context/max | 6000 / 30k | 6000 / 30k | already agree, unchanged |

Vertex's binary_NEPC sizing changes to match gpt. That is the one real behavior
change here. Longitudinal sizing is untouched in both trees.

`gap_chars` is **not** part of either profile: grepping every `build_snippet`/
`merge_windows` call site in both trees shows dfci_gpt's `merge_windows` is only
ever called from inside `build_snippet`, always at its default (300) — no live
dfci_gpt path passes another value. vertex_ai forks the same default to 80 and
applies it uniformly across every vertex task, not per-task (vertex's
`longitudinal_helpers.py` imports `build_snippet` from vertex's own
`llm_helpers.py`, same default throughout). So `gap_chars` is a per-*provider*
constant that happens to be applied uniformly, not a per-task tuning knob —
it does not belong on `SnippetProfile` alongside the genuinely task-scoped
`context_chars`/`max_chars`/`payload_max_chars`. It lives as its own top-level
`SNIPPET_GAP_CHARS = 300` constant in `preprocessing/config.py` (gpt wins → 300),
and `build_snippet`'s signature is unchanged from the original
(`context_chars`/`max_chars` keyword args only, no `gap_chars` and no `profile`
object) — `merge_windows` picks up `SNIPPET_GAP_CHARS` as its own default.

#### Resolution

Named profiles in the shared config, selected explicitly at the call site:

```python
# preprocessing/config.py — one shared module, two named profiles
SNIPPET_GAP_CHARS = 300  # per-provider, not per-task; see above

SNIPPET_PROFILES = {
    "binary_nepc":  SnippetProfile(context_chars=750,  max_chars=300_000,
                                   payload_max_chars=300_000),
    "longitudinal": SnippetProfile(context_chars=6000, max_chars=30_000,
                                   payload_max_chars=60_000),
}
```

Overridable via CLI flag and notebook parameter. The dfci_gpt comments
explaining *why* 300000 was chosen (128k-token models at ~4 chars/token ≈ 500k
input chars, budget ~300k for snippets to leave room for the system prompt and
JSON output) carry over onto the profile definitions, as does the
longitudinal note on 60k chars ≈ 15k tokens per chunk.

Net improvement over today: sizing stops being an implicit module-level shadow
that silently rebinds a shared function's keyword defaults, and becomes an
explicit named argument. Which profile a task uses is readable without tracing
the import graph.

---

## 4. Provider adapter

```python
# providers/base.py
class Provider(Protocol):
    name: str
    default_model: str

    def build_client(self): ...

    def call_with_retry(self, client, model_name, messages, max_retries=3):
        """Return (response_text, error_string). Errors are returned, never raised.

        `messages` is OpenAI-style [{"role": ..., "content": ...}].
        Adapters translate as needed.
        """
```

```python
# providers/__init__.py
def get_provider(name):
    """Lazily import the adapter so only the selected SDK is loaded."""
```

Lazy import preserves the existing deferred-import / `*_IMPORT_ERROR` pattern:
an Azure run never imports `google-cloud-aiplatform`, and vice versa. Neither
SDK is needed to run preprocessing at all.

`parse_json_response` moves to `providers/response.py` — identical in both trees
and genuinely provider-independent (it json-loads, then falls back to a
`{...}`/`[...]` regex extraction).

Runners gain `--provider {dfci_gpt,vertex_ai}`. `--model` defaults to the
selected provider's `default_model` (`gpt-4o` / `gemini-2.0-flash-001`) instead
of a module-level `DEFAULT_MODEL_NAME`.

---

## 5. Splitting the two mixed scripts

`gleason_score/extract_gleason_timeline.py` (414 lines) and
`longitudinal_NEPC/extract_avpc_nepc_timeline.py` (440) each do note collection,
LLM calls, and timeline assembly in one file. Both have the same shape, so both
split identically along existing function boundaries:

| Current symbol | Destination |
|---|---|
| `TRIGGER_REGEX` | `preprocessing/cli/collect_*_notes.py` |
| `load_notes`, `filter_note_types`, `group_patient_snippets` calls | `preprocessing/cli/collect_*_notes.py` |
| `SYSTEM_PROMPT` | `tasks/*/prompts.py` |
| `extract_patient` | `tasks/*/build_*_timeline.py` |
| `raw_rows_from_findings`, `append_rows` | `tasks/*/build_*_timeline.py` |
| `build_timeline`, `_to_int` / `_to_numeric_scalar` | `tasks/*/build_*_timeline.py` |
| `parse_args`, `run`, `main` | split across both, per flag |

The collection CLI writes a snippet artifact; the runner reads it. This gives
all four tasks the same two-phase shape `binary_NEPC` already has
(compile snippets → run LLM), which is what lets one notebook template serve
every task.

`longitudinal_NEPC` currently derives its triggers as
`{k: NEPC_TRIGGER_REGEX[k] for k in ("nepc", "avpc")}` — that subsetting of the
shared `TRIGGER_REGEX` is preserved, now importing from
`preprocessing/triggers.py`.

Keep the 5-line `append_rows` docstring from the vertex copy of
`extract_avpc_nepc_timeline.py` (it explains why polars requires the in-memory
CSV + plain-file-append workaround). This is documentation the gpt copy lacks;
the conflict rule is about behavior, and taking it loses nothing.

---

## 6. Notebooks

Four notebooks, one template, provider-toggled via subprocess — matching the
style currently in use.

```python
PROVIDER = "vertex_ai"          # or "dfci_gpt"

MODEL_NAME = {
    "dfci_gpt":  "gpt-4o",
    "vertex_ai": "gemini-2.5-flash-lite",
}[PROVIDER]

RUN_ENV = build_run_env(PROVIDER)   # Vertex project/location only when needed

# --- preprocessing: provider-independent ---
run_command([PYTHON, "preprocessing/cli/compile_patient_snippets.py",
             "--notes-csv", NOTES_CSV_PATH,
             "--output-path", SNIPPET_BUNDLE_PATH,
             "--max-notes-per-patient", str(MAX_NOTES_PER_PATIENT),
             "--scan-workers", str(SCAN_WORKERS)])

# --- LLM: one runner, provider as a flag ---
run_command([PYTHON, "tasks/binary_NEPC/run_NEPC_classifier.py",
             "--provider", PROVIDER,
             "--model", MODEL_NAME,
             "--snippets-path", SNIPPET_BUNDLE_PATH,
             "--output-dir", OUTPUT_DIR], env=RUN_ENV)
```

Conventions preserved from the current notebooks:

- every `RUN_*` step toggle defaults to `False`
- each command is built and **printed** in one cell, executed in the next
- the path-existence check cell before any run
- the `--retry-failures` cell that strips `--overwrite` and reruns failures
- `MRNS` / `MRN_FILE` / `RAW_TEXT_PATHS` parameter block at the top

`cancer_stage`, `gleason_score`, and `longitudinal_NEPC` get notebooks on this
template; they have none today.

---

## 7. Import bootstrap

All 8 scripts currently do:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

Replaced by a `pyproject.toml` + `pip install -e .`, keeping a minimal
`_bootstrap.py` shim so scripts still run directly on the cluster without an
install step. `from shared.llm_helpers import ...` becomes
`from preprocessing.notes import ...` etc.

---

## 8. Execution steps

1. Scaffold `preprocessing/`, `providers/`, `tasks/`, `notebooks/`, `tests/`.
2. Split `dfci_gpt/shared/llm_helpers.py` at the `# LLM client` marker (L907)
   using the §1.5 section map.
3. Write `providers/base.py`, `providers/response.py`, `providers/dfci_gpt.py`
   (incl. `_retry_after_seconds` / `_backoff_sleep`), `providers/vertex_ai.py`.
4. Move the identical files (`utils.py`, `longitudinal_helpers.py`,
   `snippet_bundle.py`, `compile_prostate_notes.py`,
   `compile_text_for_LLM_review.py`), rewriting imports.
5. Move task runners + prompts into `tasks/`; add `--provider`.
6. Split the gleason and longitudinal scripts per §5.
7. Apply the §3 gpt-wins fixes; add `SNIPPET_PROFILES`.
8. Replace the `sys.path` bootstrap with `pyproject.toml` + shim.
9. Write the four notebooks from the §6 template.
10. Run verification (§9).
11. Delete `dfci_gpt/` and `vertex_ai/`; rewrite root `README.md` and `CLAUDE.md`.

Steps 1–4 are the load-bearing ones; 5–7 are mechanical once the seam exists.

---

## 9. Verification

**These pipelines cannot be run end-to-end in this environment.** They read
cluster-only paths (`/data/gusev/USERS/jpconnor/...`), require Azure AD or GCP
Application Default Credentials, and the repo has no existing test suite.

What will be verified locally:

| Check | Method |
|---|---|
| preprocessing is API-free | `tests/test_no_api_imports.py` — AST-walk every module under `preprocessing/`, assert no import of `openai`, `azure`, `vertexai`, `google.cloud`. Fails loudly if API code creeps back. |
| CLIs run without SDKs | `--help` on all 6 preprocessing CLIs in an env with neither provider SDK installed |
| no cross-SDK imports | `get_provider("dfci_gpt")` then assert `"vertexai" not in sys.modules`, and the mirror |
| nothing dropped in the split | `tests/test_symbol_coverage.py` — AST symbol set of the original `llm_helpers.py` vs the union of new modules |
| snippet pipeline correctness | `tests/test_snippet_pipeline.py` — synthetic notes through `clean_note → find_trigger_matches → merge_windows → build_snippet → build_patient_snippets`, asserting the `binary_nepc` profile reproduces current dfci_gpt output |
| serial/parallel equivalence | `scan_note_candidates(max_workers=1)` vs `max_workers=4` produce identical ranked output |

**Not verifiable here, and I will report it as such rather than claim success:**
a real cluster run against OncDRS data, live Azure/Vertex API calls, and
therefore any end-to-end confirmation that label output is unchanged. Suggested
acceptance check on your side: run one task with `--limit-mrns` on a small MRN
list under both providers and diff against current output.

---

## 10. Scope notes

- **"New repo here"** is implemented as a new top-level directory in this
  repository, per your selection. `preprocessing/` will be independently
  importable with its own `pyproject.toml`, so promoting it to a separate git
  repo later is a directory move, not a rewrite.
- **Git state:** everything is currently uncommitted on `main`, with a large
  pending rename already showing as deletes + untracked files. This refactor
  moves nearly every file in the repo. Committing the current state first is
  recommended as a rollback point. No commit or push will happen without your
  say-so.
- `.ipynb_checkpoints/` artifacts currently tracked in `dfci_gpt/` should be
  added to `.gitignore` rather than carried over.
