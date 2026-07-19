"""Shared plumbing for the longitudinal, patient-chunked LLM extraction pipelines.

Both pipelines (Gleason timeline, AVPC/NEPC criteria timeline) collect
trigger-bearing notes, de-duplicate them per patient, pack them into one or more
payload-sized chunks, and make one LLM call per chunk. The generic LLM client /
retry / JSON / note-loading helpers are reused from the NEPC classifier; this
module adds the snippet iteration, patient chunking, and small extraction
utilities those pipelines share.
"""

import hashlib
import math
import re

import polars as pl

# Reused, trigger-agnostic plumbing from the shared LLM helpers.
from shared.llm_helpers import (  # noqa: E402,F401
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_NAME,
    PROSTATE_TEXT_CSV,
    TRIGGER_REGEX as NEPC_TRIGGER_REGEX,
    build_client,
    build_snippet,
    call_with_retry,
    clean_note,
    load_notes,
    load_selected_mrns,
    parse_json_response,
    to_iso_date,
)

SNIPPET_CONTEXT_CHARS = 6000
SNIPPET_MAX_CHARS = 30000
# Max snippet chars packed into one LLM call (one chunk). ~60k chars ≈ 15k tokens,
# leaving ample room under a 128k-token model for the system prompt + JSON output.
DEFAULT_PAYLOAD_MAX_CHARS = 60000


def flatten_ws(value):
    """Collapse tabs/newlines/whitespace runs to single spaces for safe TSV storage.

    Free-text fields (verbatim quotes) can contain tabs or newlines that would shift
    columns / split rows in a tab-separated file; flattening them keeps the TSV aligned.
    """
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def find_matches(text, trigger_regex):
    """Return sorted (label, start, end) tuples for every trigger hit in text."""
    matches = []
    for label, pattern in trigger_regex.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((label, match.start(), match.end()))
    return sorted(matches, key=lambda item: (item[1], item[2]))


def filter_note_types(notes_df, note_types):
    """Restrict notes to the given NOTE_TYPE values (case-insensitive).

    `note_types` is a list/iterable (e.g. ["Pathology"]) or falsy to keep all.
    """
    if not note_types:
        return notes_df
    wanted = {str(t).strip().lower() for t in note_types}
    return notes_df.filter(pl.col("NOTE_TYPE").cast(pl.Utf8).str.to_lowercase().is_in(wanted))


def _note_uid(mrn, note_date, snippet, raw_note_id):
    """Stable per-note identifier for resumability/dedup.

    Prefers the source RAW_NOTE_ID; otherwise hashes (mrn, date, snippet head)."""
    if raw_note_id is not None and not (isinstance(raw_note_id, float) and math.isnan(raw_note_id)):
        return str(raw_note_id)
    key = f"{int(mrn)}|{note_date or ''}|{snippet[:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def iter_note_snippets(
    notes_df,
    trigger_regex,
    *,
    context_chars=SNIPPET_CONTEXT_CHARS,
    snippet_max_chars=SNIPPET_MAX_CHARS,
):
    """Yield one snippet dict per trigger-bearing note, deduplicated per patient.

    Each yielded dict has: note_uid, DFCI_MRN, note_date (ISO or None), note_type,
    trigger_categories (sorted list), snippet. Copy-forward notes with identical
    cleaned snippet text are collapsed per patient, keeping the EARLIEST note_date
    as the provenance/fallback date (so criterion-onset dates are not inflated).
    """
    if notes_df.is_empty():
        return

    # (mrn, snippet) -> chosen record, keeping the earliest note_date.
    deduped = {}
    for row in notes_df.iter_rows(named=True):
        raw_text = row.get("CLINICAL_TEXT") or ""
        note_type = row.get("NOTE_TYPE") or "Unknown"
        cleaned = clean_note(raw_text, note_type=note_type)
        if not cleaned:
            continue
        matches = find_matches(cleaned, trigger_regex)
        if not matches:
            continue
        snippet = build_snippet(
            cleaned, matches, context_chars=context_chars, max_chars=snippet_max_chars
        )
        if not snippet:
            continue
        mrn = int(row["DFCI_MRN"])
        note_date = to_iso_date(row.get("EVENT_DATE"))
        record = {
            "note_uid": _note_uid(
                mrn, note_date, snippet, row.get("RAW_NOTE_ID")
            ),
            "DFCI_MRN": mrn,
            "note_date": note_date,
            "note_type": note_type,
            "trigger_categories": sorted({m[0] for m in matches}),
            "snippet": snippet,
        }
        key = (mrn, snippet)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
        else:
            # Keep the earliest available date as the canonical provenance.
            existing_date = existing["note_date"] or ""
            new_date = note_date or ""
            if new_date and (not existing_date or new_date < existing_date):
                existing["note_date"] = note_date
                existing["note_type"] = note_type
                existing["note_uid"] = record["note_uid"]

    yield from deduped.values()


def resolve_date(stated_date, note_date):
    """Resolve an event date: a valid stated date wins, else fall back to note_date.

    Returns (resolved_iso_or_None, date_source) where date_source is
    "stated" | "note_date" | "unknown". Both inputs are normalized through
    to_iso_date, so NaN / empty / unparseable values never leak through (and a
    missing note_date can't crash downstream string-sorted aggregation).
    """
    iso = to_iso_date(stated_date)
    if iso:
        return iso, "stated"
    note_iso = to_iso_date(note_date)
    if note_iso:
        return note_iso, "note_date"
    return None, "unknown"


def group_patient_snippets(
    notes_df,
    trigger_regex,
    *,
    context_chars=SNIPPET_CONTEXT_CHARS,
    snippet_max_chars=SNIPPET_MAX_CHARS,
    payload_max_chars=DEFAULT_PAYLOAD_MAX_CHARS,
):
    """Group deduped per-note snippets by patient into payload-sized chunks.

    Returns {mrn: [chunk, ...]} where each chunk is a list of snippet records
    (same dicts iter_note_snippets yields). Snippets are de-duplicated per patient,
    ordered chronologically (oldest first), then packed greedily up to
    `payload_max_chars`. Chunking — rather than capping — means NO snippet is
    dropped, so earliest-occurrence dates and rare findings survive even for
    heavily-documented patients; the number of LLM calls is one per chunk
    (one for most patients, a few for outliers) instead of one per note.
    """
    by_mrn = {}
    for rec in iter_note_snippets(
        notes_df,
        trigger_regex,
        context_chars=context_chars,
        snippet_max_chars=snippet_max_chars,
    ):
        by_mrn.setdefault(rec["DFCI_MRN"], []).append(rec)

    patient_chunks = {}
    for mrn, recs in by_mrn.items():
        recs.sort(key=lambda r: (r["note_date"] or "9999-99-99"))
        chunks, current, current_len = [], [], 0
        for rec in recs:
            slen = len(rec["snippet"])
            if current and current_len + slen > payload_max_chars:
                chunks.append(current)
                current, current_len = [], 0
            current.append(rec)
            current_len += slen
        if current:
            chunks.append(current)
        patient_chunks[mrn] = chunks
    return patient_chunks


def derive_grade_group(primary, secondary):
    """Derive ISUP Grade Group (1-5) from Gleason primary/secondary patterns."""
    try:
        p, s = int(primary), int(secondary)
    except (TypeError, ValueError):
        return None
    if not (1 <= p <= 5 and 1 <= s <= 5):
        return None
    total = p + s
    if total <= 6:
        return 1
    if p == 3 and s == 4:
        return 2
    if p == 4 and s == 3:
        return 3
    if total == 8:
        return 4
    if total in (9, 10):
        return 5
    return None
