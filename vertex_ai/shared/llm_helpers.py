import gzip
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
try:
    import ijson
except ImportError:
    ijson = None

try:
    import google.api_core.exceptions as gcp_exceptions
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    VERTEX_IMPORT_ERROR = None
except ImportError as error:
    gcp_exceptions = None
    vertexai = None
    GenerativeModel = None
    GenerationConfig = None
    VERTEX_IMPORT_ERROR = error


from shared.utils import clean_note  # noqa: E402


# Paths
DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "LLM_ANNOTATIONS_DATA_PATH",
        os.environ.get("CAIA_COMPASS_DATA_PATH", "/data/gusev/USERS/jpconnor/data/LLM_annotations/"),
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "BINARY_NEPC_OUTPUT_DIR",
        os.environ.get(
            "CAIA_COMPASS_NEPC_CLASSIFIER_OUTPUT_DIR",
            str(DEFAULT_DATA_PATH / "LLM_NEPC_labels"),
        ),
    )
)
DEFAULT_RAW_TEXT_PATHS = (
    Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2024_03/"),
    Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_03/"),
    Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_11/"),
)
NOTE_BUNDLE_FILENAME = "LLM_NEPC_classifier_note_bundle.json.gz"
PROSTATE_TEXT_CSV = DEFAULT_DATA_PATH / "prostate_text_data.csv"

NOTE_BUNDLE_COLUMNS = (
    "DFCI_MRN",
    "EVENT_DATE",
    "NOTE_TYPE",
    "CLINICAL_TEXT",
    "RAW_SOURCE_FILE",
    "RAW_NOTE_ID",
    "RPT_DATE",
    "RPT_TYPE",
    "SOURCE_STR",
    "PROC_DESC_STR",
    "ENCOUNTER_TYPE_DESC_STR",
)


# Vertex AI
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "gusevlabllm")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
DEFAULT_MODEL_NAME = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash-001")


# Triggers — combined NEPC + AVPC + biomarker terms
TRIGGER_REGEX = {
    "nepc": (
        r"\b(?:"
        r"neuroendocrine|neuro-endocrine|nepc|t-nepc|"
        r"small[\s-]?cell(?:\s+carcinoma)?|scpc|scnc|oat[\s-]?cell|"
        r"small[- ]cell\s+neuroendocrine\s+carcinoma|"
        r"histolog(?:ic|ical)\s+transform(?:ation|ed|ing)|"
        r"transform(?:ation|ed|ing)(?:\s+(?:to|into))?|"
        r"transdifferentiat(?:e|ed|ion|ing)|dedifferentiat(?:e|ed|ion|ing)|"
        r"lineage\s+plasticity|treatment[\s-]?emergent\s+neuroendocrine|"
        r"synaptophysin|chromogranin(?:\s+a)?|cd56|neuron[- ]specific\s+enolase|nse"
        r")\b"
    ),
    "avpc": (
        r"\b(?:"
        r"aggressive[\s-]?variant|avpc|anaplastic|variant\s+crpc|androgen[- ]indifferent|"
        r"visceral\s+met(?:astases|astasis|astatic)?|"
        r"liver\s+met(?:astases|astasis|astatic)?|hepatic\s+met(?:astases|astasis|astatic)?|"
        r"lung\s+met(?:astases|astasis|astatic)?|pulmonary\s+met(?:astases|astasis|astatic)?|"
        r"adrenal\s+met(?:astases|astasis|astatic)?|brain\s+met(?:astases|astasis|astatic)?|"
        r"pleural\s+met(?:astases|astasis|astatic)?|peritoneal\s+met(?:astases|astasis|astatic)?|"
        r"lytic\s+(?:bone|lesion)|predominantly\s+lytic|osseous\s+lytic|destructive\s+bone\s+lesion|"
        r"bulky\s+(?:lymphadenopathy|adenopathy|nodal|nodes?|pelvic\s+mass|prostate\s+mass)|"
        r"large\s+(?:pelvic|prostatic)\s+mass|"
        r"low\s+psa|disproportionately\s+low\s+psa|psa\s+discordant|"
        r"(?:high[- ]volume|extensive|diffuse|innumerable)\s+(?:bone|osseous)\s+met(?:astases|astatic)?|"
        r"bombesin|grp|cea|ldh|hypercalc(?:emia|aemia)|"
        r"castration[- ]resistant|androgen[- ]independent|rapidly?\s+progress(?:ion|ive)|"
        r"refractory\s+to\s+adt|despite\s+adt"
        r")\b"
    ),
    "biomarker": (
        r"\b(?:"
        r"brca1|brca2|atm|cdk12|palb2|"
        r"hrd|hrr|ddr|homologous\s+recombination|dna\s+damage\s+repair|"
        r"msi[- ]h(?:igh)?|mmr|mismatch\s+repair|msh2|msh6|mlh1|pms2|"
        r"tumor\s+mutational\s+burden|tmb"
        r")\b"
    ),
    "non_prostate_primary": (
        r"\b(?:"
        # Lung
        r"nsclc|sclc|non[- ]small[- ]cell\s+lung|lung\s+adenocarcinoma|lung\s+(?:cancer|carcinoma)|"
        # GI
        r"colorectal|colon\s+(?:cancer|carcinoma)|rectal\s+(?:cancer|carcinoma)|"
        r"pancreatic\s+(?:cancer|carcinoma|adenocarcinoma)|gastric\s+(?:cancer|carcinoma)|"
        r"esophageal\s+(?:cancer|carcinoma)|hepatocellular\s+carcinoma|hcc|"
        # GU (non-prostate)
        r"urothelial\s+(?:cancer|carcinoma)|bladder\s+(?:cancer|carcinoma)|"
        r"renal\s+cell\s+carcinoma|rcc|kidney\s+(?:cancer|carcinoma)|"
        # Heme
        r"lymphoma|leukemia|multiple\s+myeloma|"
        # Other solid
        r"melanoma|glioblastoma|head\s+and\s+neck\s+(?:cancer|carcinoma|squamous)|"
        r"breast\s+(?:cancer|carcinoma)|"
        # Multi-primary phrases
        r"second\s+primary|synchronous\s+primary|metachronous\s+primary|"
        r"history\s+of\s+(?:lung|colon|colorectal|breast|bladder|kidney|renal|pancreatic|gastric|esophageal|melanoma|lymphoma|leukemia)"
        r")\b"
    ),
}


CLINICAL_SAFETY_CONTEXT = """

IMPORTANT CONTEXT: All notes below are de-identified clinical oncology documentation being
processed for structured data extraction as part of an IRB-approved medical research study
(institutional review board approved protocol). This is professional medical documentation
written by physicians, not patient-generated content. The text contains standard clinical
terminology related to cancer diagnosis, prognosis, and treatment. References to disease
outcomes, end-of-life care, self-harm assessment, psychiatric history, substance use, anatomy,
or patient distress are routine components of oncology and medical records and should be
processed as clinical data. No content in these notes constitutes harmful, dangerous, or
inappropriate material - it is standard-of-care medical documentation.
"""


CLASSIFY_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved prostate cancer research study.

You will receive a JSON payload of de-identified clinical note snippets for a single patient.
Each snippet was selected because it contains language relevant to one of:
- neuroendocrine prostate cancer (NEPC)
- aggressive-variant prostate cancer (AVPC)
- platinum-relevant molecular biomarkers
- a non-prostate primary cancer (separate annotation)

## YOUR TASK
1. Classify the patient into ONE primary bucket and report supporting evidence,
   applying this precedence (return the highest-precedence bucket documented):

   a. nepc — chart documents ANY of the following on a prostate-derived specimen or in
      a prostate-cancer patient's oncologic documentation:
        - neuroendocrine or small-cell prostate carcinoma diagnosis
        - histologic transformation from adenocarcinoma to neuroendocrine / small-cell
        - neuroendocrine features or neuroendocrine differentiation (focal, partial, or
          "with NE features" / "component of" all qualify)
        - positive neuroendocrine IHC markers (synaptophysin, chromogranin, CD56, NSE,
          INSM1) on a prostate-derived specimen
      Any documented neuroendocrine feature is sufficient for `nepc` — do NOT downgrade
      to `avpc` because the wording is hedged.
   b. avpc — chart documents aggressive-variant or anaplastic prostate cancer language, OR
      satisfies one or more Aparicio aggressive-variant criteria:
        C1 small-cell histology
        C2 visceral metastatic pattern — metastasis to lung, adrenal, brain, pleura,
           or peritoneum. Liver / hepatic metastases alone do NOT qualify as C2; C2
           requires at least one of the qualifying visceral sites above. When C2 is
           set, also populate `visceral_met_pattern`:
             "visceral_only"     — qualifying visceral mets with NO concurrent bone mets
             "visceral_and_bone" — qualifying visceral mets WITH concurrent bone mets
           When C2 is NOT set, `visceral_met_pattern` must be "none".
        C3 predominantly lytic bone metastases
        C4 bulky disease — restricted to: (a) bulky lymphadenopathy / nodal disease,
           OR (b) prostate or pelvic mass with a documented measurement of at least
           5 cm. Generic wording like "large pelvic mass" or "bulky disease" WITHOUT
           a specific ≥ 5 cm measurement does NOT qualify for C4.
        C5 low PSA with high-volume disease
        C6 neuroendocrine markers / elevated CEA or LDH / hypercalcemia (when explicit)
        C7 rapid progression to castration-resistant or androgen-independent disease
   c. biomarker — chart documents a QUALIFYING SOMATIC (tumor) biomarker. The qualifying
      set is restricted to: BRCA1, BRCA2, PALB2. Only these three genes cause the
      primary bucket to be `biomarker` and `has_biomarker` to be true.
      Only count findings from tumor/somatic testing (e.g., tumor NGS, OncoPanel, FoundationOne,
      Tempus, MSK-IMPACT, ctDNA/liquid biopsy of tumor). Do NOT count germline findings — exclude
      results from germline panels, hereditary / familial testing, blood/saliva germline assays,
      or variants explicitly labeled "germline". If a variant is ambiguous between germline and
      somatic, do not set `has_biomarker = true`.
      Other somatic biomarkers (ATM, CDK12, HRD/HRR, DDR pathway, MSI-H, MMR-deficient,
      TMB-high, PTEN, TP53, RB1, AR variants, SPOP, etc.) must still be RECORDED in
      `biomarker_genes` when documented, but they do NOT by themselves set
      `has_biomarker = true` or change the primary bucket.
   d. conventional — none of the above.

   PRECEDENCE IS STRICT: if NEPC criteria are met, the primary bucket is `nepc` even when
   AVPC criteria (C1–C7) are ALSO met. `avpc` is only chosen when NEPC criteria are absent.

2. SEPARATELY, flag whether the chart documents a NON-PROSTATE PRIMARY cancer
   (synchronous or metachronous, e.g., NSCLC/SCLC, colorectal, urothelial/bladder,
   renal cell, pancreatic, gastric, hepatocellular, lymphoma, melanoma, head and neck,
   breast). This annotation is INDEPENDENT of the primary bucket — a patient classified
   as `nepc` can still have `has_non_prostate_primary = true` if both are documented.

3. SEPARATELY, set `has_molecular_avpc = true` when the chart documents SOMATIC (tumor)
   alterations in AT LEAST TWO of the following three genes: PTEN, TP53, RB1. A single
   alteration in one of these genes alone is NOT sufficient. This annotation is fully
   INDEPENDENT of the primary bucket and of `has_avpc` — setting `has_molecular_avpc`
   does NOT set `has_avpc`, does NOT add a C-criterion to `avpc_criteria`, and does NOT
   change `primary_label`. Apply the same somatic-only rule as the biomarker bucket:
   exclude germline findings, ambiguous germline/somatic findings, and any variant
   explicitly labeled "germline". PTEN / TP53 / RB1 alterations must still be listed
   in `biomarker_genes` when documented, regardless of whether `has_molecular_avpc`
   is set.

## RULES
- Use only the snippets provided. Do not infer beyond documented evidence.
- Suspicion, screening, planned testing, pending stains, and clinical-trial eligibility
  language do NOT establish a diagnosis or biomarker finding by themselves.
- Pathology is most authoritative for histology. Imaging is most authoritative for
  metastatic pattern.
- Read every pathology snippet end-to-end. Neuroendocrine findings, IHC panels, and
  small-cell histology routinely live in the microscopic / addendum sections and must
  not be missed — missing a pathology NEPC finding is the most common failure mode.
- If a pathology report documents NE features, set `has_nepc = true` and
  `primary_label = "nepc"` even if clinician notes still describe the disease as AVPC
  or adenocarcinoma.
- Quotes must be verbatim. Prefer substantive quotes that preserve surrounding clinical
  context (roughly 40–120 words each). Include the sentence(s) on either side of the
  key finding when they clarify specimen source, timing, or diagnostic certainty.
- For `has_non_prostate_primary`: only set true when the chart documents the patient
  currently has or previously had a non-prostate primary cancer. Do NOT count family
  history, differential-diagnosis mentions, ruled-out workup, or "no history of other
  malignancies" statements. List specific cancer types in `non_prostate_primary_types`.
- For `biomarker_genes`: list EVERY somatic biomarker / gene alteration documented in
  the chart (e.g., ["BRCA2", "ATM", "TMB-high", "TP53"]), regardless of whether it
  qualifies for the `biomarker` bucket or the `has_molecular_avpc` flag. `has_biomarker`
  and `primary_label = "biomarker"` are gated on the qualifying set (BRCA1, BRCA2, PALB2)
  only. `has_molecular_avpc` is gated on ≥ 2 somatic alterations in {PTEN, TP53, RB1}.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "primary_label": "nepc | avpc | biomarker | conventional",
  "has_nepc": true | false,
  "has_avpc": true | false,
  "has_biomarker": true | false,
  "has_molecular_avpc": true | false,
  "biomarker_genes": ["BRCA2"],
  "avpc_criteria": ["C1", "C2"],
  "visceral_met_pattern": "visceral_only | visceral_and_bone | none",
  "has_non_prostate_primary": true | false,
  "non_prostate_primary_types": ["NSCLC", "colorectal"],
  "supporting_quotes": ["<verbatim quote>"],
  "supporting_quote_dates": ["YYYY-MM-DD"],
  "confidence": "high | medium | low",
  "rationale": "<1-2 sentences>"
}
"""


# MRN parsing
def parse_mrn_values(values):
    mrns = set()
    for value in values:
        if pd.isna(value):
            continue
        for token in re.split(r"[\s,|]+", str(value).strip()):
            if not token:
                continue
            mrn = pd.to_numeric(token, errors="coerce")
            if pd.notna(mrn):
                mrns.add(int(mrn))
    return mrns


def load_selected_mrns(mrns_arg=None, mrn_file=None):
    selected = set()
    if mrns_arg:
        selected.update(parse_mrn_values([mrns_arg]))
    if mrn_file:
        mrn_file = Path(mrn_file)
        suffix = mrn_file.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            mrn_df = pd.read_csv(mrn_file, sep=sep, low_memory=False)
            if "DFCI_MRN" in mrn_df.columns:
                selected.update(parse_mrn_values(mrn_df["DFCI_MRN"]))
            elif not mrn_df.empty:
                selected.update(parse_mrn_values(mrn_df.iloc[:, 0]))
        else:
            with open(mrn_file, "r", encoding="utf-8") as handle:
                selected.update(parse_mrn_values(handle.readlines()))
    return selected or None


def normalize_mrn_column(df):
    if df.empty or "DFCI_MRN" not in df.columns:
        return df
    work = df.copy()
    work["DFCI_MRN"] = pd.to_numeric(work["DFCI_MRN"], errors="coerce")
    work = work.dropna(subset=["DFCI_MRN"])
    work["DFCI_MRN"] = work["DFCI_MRN"].astype(int)
    return work


# Note text utilities
def basic_clean_text(text):
    cleaned = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", " ")
        .replace("\xa0", " ")
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def deduplicate_texts(text_entries):
    seen = set()
    deduped = []
    for entry in text_entries:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def to_iso_date(value):
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


# Raw / bundle loaders
def infer_note_type_from_filename(path):
    name = Path(path).name.lower()
    if "imaging" in name:
        return "Imaging"
    if "prognote" in name or "progress" in name or "clinic" in name:
        return "Clinician"
    if "pathology" in name or re.search(r"(^|[-_])path(?:[-_.]|$)", name):
        return "Pathology"
    return None


def discover_raw_text_files(raw_text_paths):
    discovered = []
    seen = set()
    for raw_text_path in raw_text_paths:
        raw_text_path = Path(raw_text_path)
        if not raw_text_path.exists():
            continue
        for path in sorted(raw_text_path.rglob("*.json")):
            note_type = infer_note_type_from_filename(path)
            if note_type is not None and str(path) not in seen:
                seen.add(str(path))
                discovered.append((path, note_type))
    return discovered


def extract_raw_docs(payload):
    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("docs"), list):
            return response["docs"]
        if isinstance(payload.get("docs"), list):
            return payload["docs"]
    if isinstance(payload, list):
        return payload
    return []


def iter_raw_docs_from_file(path):
    if ijson is None:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from extract_raw_docs(payload)
        return

    with open(path, "r", encoding="utf-8") as handle:
        for prefix in ("response.docs.item", "docs.item", "item"):
            handle.seek(0)
            try:
                iterator = ijson.items(handle, prefix)
                first = next(iterator, None)
            except ijson.JSONError:
                continue
            if first is None:
                continue
            yield first
            yield from iterator
            return


def build_raw_note_row(note, note_type, source_file):
    mrn = pd.to_numeric(note.get("DFCI_MRN"), errors="coerce")
    if pd.isna(mrn):
        return None
    text_entries = [v for k, v in note.items() if "TEXT" in str(k).upper()]
    text = basic_clean_text(" ".join(deduplicate_texts(text_entries)))
    if not text:
        return None
    return {
        "DFCI_MRN": int(mrn),
        "EVENT_DATE": note.get("EVENT_DATE") or note.get("RPT_DATE"),
        "NOTE_TYPE": note_type,
        "CLINICAL_TEXT": text,
        "RAW_SOURCE_FILE": Path(source_file).name,
        "RAW_NOTE_ID": note.get("id"),
        "RPT_DATE": note.get("RPT_DATE"),
        "RPT_TYPE": note.get("RPT_TYPE"),
        "SOURCE_STR": note.get("SOURCE_STR"),
        "PROC_DESC_STR": note.get("PROC_DESC_STR"),
        "ENCOUNTER_TYPE_DESC_STR": note.get("ENCOUNTER_TYPE_DESC_STR"),
    }


def resolve_raw_text_paths(raw_text_paths_arg=None):
    if not raw_text_paths_arg:
        return list(DEFAULT_RAW_TEXT_PATHS)
    seen, paths = set(), []
    for path in raw_text_paths_arg:
        normalized = Path(path)
        key = str(normalized)
        if key not in seen:
            seen.add(key)
            paths.append(normalized)
    return paths


def load_raw_text_notes(raw_text_paths, selected_mrns):
    if selected_mrns is None:
        raise ValueError("Raw text mode requires --mrns or --mrn-file.")
    raw_files = discover_raw_text_files(raw_text_paths)
    if not raw_files:
        joined = ", ".join(str(p) for p in raw_text_paths)
        raise FileNotFoundError(f"No supported raw JSON note files found under: {joined}")
    rows = []
    for file_path, note_type in raw_files:
        for note in iter_raw_docs_from_file(file_path):
            mrn = pd.to_numeric(note.get("DFCI_MRN"), errors="coerce")
            if pd.isna(mrn) or int(mrn) not in selected_mrns:
                continue
            row = build_raw_note_row(note, note_type, file_path)
            if row is not None:
                rows.append(row)
    df = normalize_mrn_column(pd.DataFrame(rows))
    if df.empty:
        raise ValueError("No raw notes were found for the requested MRNs.")
    return df


def write_note_bundle(path, note_df, *, raw_text_paths=None, selected_mrns=None):
    if note_df.empty:
        standardized = pd.DataFrame(columns=list(NOTE_BUNDLE_COLUMNS))
    else:
        keep_cols = [c for c in NOTE_BUNDLE_COLUMNS if c in note_df.columns]
        standardized = note_df[keep_cols].copy()
        if "EVENT_DATE" in standardized.columns:
            standardized["EVENT_DATE"] = pd.to_datetime(
                standardized["EVENT_DATE"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        standardized = standardized.sort_values(
            ["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last"
        )
    serializable = standardized.copy().astype(object).where(pd.notna(standardized), None)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_mrn_count": len(selected_mrns) if selected_mrns is not None else None,
        "patient_count": int(standardized["DFCI_MRN"].nunique()) if not standardized.empty else 0,
        "note_count": int(len(standardized)),
        "raw_text_paths": [str(p) for p in raw_text_paths] if raw_text_paths else None,
        "notes": serializable.to_dict(orient="records") if not standardized.empty else [],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def write_notes_csv(path, note_df):
    if note_df.empty:
        standardized = pd.DataFrame(columns=list(NOTE_BUNDLE_COLUMNS))
    else:
        keep_cols = [c for c in NOTE_BUNDLE_COLUMNS if c in note_df.columns]
        standardized = note_df[keep_cols].copy()
        if "EVENT_DATE" in standardized.columns:
            standardized["EVENT_DATE"] = pd.to_datetime(
                standardized["EVENT_DATE"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        standardized = standardized.sort_values(
            ["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last"
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)
    return standardized


def load_note_bundle(path, selected_mrns=None):
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Note bundle not found: {bundle_path}")
    with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get("notes", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    df = normalize_mrn_column(pd.DataFrame(records))
    if df.empty:
        raise ValueError(f"No note rows in bundle: {bundle_path}")
    if selected_mrns is not None:
        df = df.loc[df["DFCI_MRN"].isin(selected_mrns)].copy()
        if df.empty:
            raise ValueError("No notes after MRN filter.")
    return df


def load_notes_csv(csv_path, selected_mrns=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prostate notes CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if "CLINICAL_TEXT" not in df.columns:
        raise ValueError(f"Prostate notes CSV missing CLINICAL_TEXT column: {csv_path}")
    df = normalize_mrn_column(df)
    if df.empty:
        raise ValueError(f"No note rows in CSV: {csv_path}")
    if selected_mrns is not None:
        df = df.loc[df["DFCI_MRN"].isin(selected_mrns)].copy()
        if df.empty:
            raise ValueError("No notes after MRN filter.")
    return df


def load_notes(*, csv_path=None, bundle_path=None, raw_text_paths=None, selected_mrns=None):
    """Load prostate notes for the LLM pipelines.

    Precedence: an explicitly-provided bundle that exists > the compiled
    prostate_text_data.csv > raw OncDRS JSONs. The CSV is the default source.
    """
    if bundle_path is not None and Path(bundle_path).exists():
        return load_note_bundle(bundle_path, selected_mrns)
    csv_path = csv_path or PROSTATE_TEXT_CSV
    if Path(csv_path).exists():
        return load_notes_csv(csv_path, selected_mrns)
    return load_raw_text_notes(resolve_raw_text_paths(raw_text_paths), selected_mrns)


# Snippet building
SNIPPET_CONTEXT_CHARS = 6000
SNIPPET_MAX_CHARS = 30000
PATIENT_PAYLOAD_MAX_CHARS = 300000


def merge_windows(windows, gap_chars=80):
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_chars:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def find_trigger_matches(text):
    matches = []
    for label, pattern in TRIGGER_REGEX.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((label, match.start(), match.end()))
    return sorted(matches, key=lambda item: (item[1], item[2]))


def build_snippet(text, matches, *, context_chars=SNIPPET_CONTEXT_CHARS, max_chars=SNIPPET_MAX_CHARS):
    if not matches:
        return ""
    windows = [(max(0, s - context_chars), min(len(text), e + context_chars)) for _, s, e in matches]
    parts = []
    for start, end in merge_windows(windows):
        snippet = text[start:end].strip()
        if not snippet:
            continue
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        parts.append(snippet)
    out = "\n\n...\n\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out


def build_patient_snippets(
    notes_df,
    *,
    max_notes_per_patient=30,
    snippet_max_chars=SNIPPET_MAX_CHARS,
    payload_max_chars=PATIENT_PAYLOAD_MAX_CHARS,
):
    """Return {mrn: [{note_date, note_type, trigger_categories, snippet}, ...]}.

    Notes without any trigger hit are dropped. Per patient, notes are ranked by
    (number of trigger categories, raw trigger count, recency) and kept until either
    `max_notes_per_patient` or the cumulative `payload_max_chars` budget is hit
    (whichever comes first), so outlier patients can't exceed the model's context window.
    """
    if notes_df.empty:
        return {}

    candidates = {}
    for row in notes_df.itertuples(index=False):
        raw_text = getattr(row, "CLINICAL_TEXT", None) or ""
        note_type = getattr(row, "NOTE_TYPE", None) or "Unknown"
        cleaned = clean_note(raw_text, note_type=note_type)
        if not cleaned:
            continue
        matches = find_trigger_matches(cleaned)
        if not matches:
            continue
        snippet = build_snippet(cleaned, matches, max_chars=snippet_max_chars)
        if not snippet:
            continue
        mrn = int(row.DFCI_MRN)
        categories = sorted({m[0] for m in matches})
        candidates.setdefault(mrn, []).append({
            "note_date": to_iso_date(getattr(row, "EVENT_DATE", None)),
            "note_type": note_type,
            "trigger_categories": categories,
            "trigger_count": len(matches),
            "snippet": snippet,
        })

    ranked = {}
    for mrn, items in candidates.items():
        items.sort(
            key=lambda c: (
                len(c["trigger_categories"]),
                c["trigger_count"],
                c["note_date"] or "",
            ),
            reverse=True,
        )
        kept = []
        used_chars = 0
        for c in items[:max_notes_per_patient]:
            snippet_len = len(c["snippet"])
            if kept and used_chars + snippet_len > payload_max_chars:
                break
            kept.append({
                "note_date": c["note_date"],
                "note_type": c["note_type"],
                "trigger_categories": c["trigger_categories"],
                "snippet": c["snippet"],
            })
            used_chars += snippet_len
        ranked[mrn] = kept
    return ranked


# LLM client — Vertex AI
def build_client():
    if VERTEX_IMPORT_ERROR is not None:
        raise ImportError(
            "Vertex AI note extraction requires google-cloud-aiplatform in the active environment."
        ) from VERTEX_IMPORT_ERROR
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    # GenerativeModel is instantiated per call; return None as a sentinel so call sites
    # that pass `client` as the first arg to call_with_retry still work unchanged.
    return None


def call_with_retry(client, model_name, messages, max_retries=3):  # noqa: ARG001 (client unused)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    model = GenerativeModel(model_name, system_instruction=system_instruction)
    generation_config = GenerationConfig(temperature=0, response_mime_type="application/json")

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                "\n\n".join(user_parts),
                generation_config=generation_config,
            )
            candidate = response.candidates[0]
            finish_name = candidate.finish_reason.name
            if finish_name in ("SAFETY", "RECITATION", "BLOCKLIST"):
                return None, "content_filter_response"
            text = candidate.content.parts[0].text.strip()
            return text, None
        except gcp_exceptions.ResourceExhausted:
            time.sleep(2 ** attempt * 5)
        except gcp_exceptions.DeadlineExceeded:
            time.sleep(2 ** attempt * 3)
        except gcp_exceptions.GoogleAPIError as error:
            body = str(error)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f"api_error: {body[:200]}"
        except Exception as error:  # noqa: BLE001
            return None, f"unexpected: {type(error).__name__}: {str(error)[:200]}"
    return None, "max_retries_exceeded"


def parse_json_response(response_text):
    if response_text is None:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", response_text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise
