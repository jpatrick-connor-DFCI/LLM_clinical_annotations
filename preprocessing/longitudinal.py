"""Shared plumbing for the longitudinal, patient-chunked LLM extraction pipelines.

All four pipelines (Gleason timeline, AVPC/NEPC criteria timeline, cancer stage,
binary NEPC) collect trigger-bearing notes and snippet them via `preprocessing`;
the longitudinal pipelines additionally de-duplicate per patient and pack
snippets into one or more payload-sized chunks for a single LLM call per chunk.
"""

import hashlib
import math
import re

import polars as pl

from preprocessing.config import SNIPPET_PROFILES
from preprocessing.notes import to_iso_date
from preprocessing.snippets import scan_note_candidates

_LONGITUDINAL_PROFILE = SNIPPET_PROFILES["longitudinal"]


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


def _candidate_to_record(candidate):
    """Convert a scan_note_candidates() candidate into the record shape dedupe/pack expect.

    scan_note_candidates (preprocessing/snippets.py) is the ProcessPoolExecutor-parallel
    per-note scan shared with the binary_nepc path; its candidates carry `mrn`/`raw_note_id`
    instead of `DFCI_MRN`/`note_uid`, so this adapts field names rather than duplicating
    the scan logic.
    """
    mrn = candidate["mrn"]
    note_date = candidate["note_date"]
    return {
        "note_uid": _note_uid(mrn, note_date, candidate["snippet"], candidate.get("raw_note_id")),
        "DFCI_MRN": mrn,
        "note_date": note_date,
        "note_type": candidate["note_type"],
        "trigger_categories": candidate["trigger_categories"],
        "snippet": candidate["snippet"],
    }


def dedupe_candidates(candidates):
    """Collapse copy-forward notes with identical cleaned snippet text per patient.

    `candidates` is {mrn: [candidate, ...]} as returned by scan_note_candidates().
    Dedup key is (mrn, snippet); the EARLIEST note_date wins as the provenance/
    fallback date (so criterion-onset dates are not inflated). This must run in the
    parent process (never per-worker) because the same snippet can be produced by
    notes scanned in different workers.

    Yields one record per kept (mrn, snippet) pair: note_uid, DFCI_MRN, note_date
    (ISO or None), note_type, trigger_categories (sorted list), snippet.
    """
    # (mrn, snippet) -> chosen record, keeping the earliest note_date.
    deduped = {}
    for mrn, items in candidates.items():
        for candidate in items:
            record = _candidate_to_record(candidate)
            key = (mrn, record["snippet"])
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = record
            else:
                # Keep the earliest available date as the canonical provenance.
                existing_date = existing["note_date"] or ""
                new_date = record["note_date"] or ""
                if new_date and (not existing_date or new_date < existing_date):
                    existing["note_date"] = record["note_date"]
                    existing["note_type"] = record["note_type"]
                    existing["note_uid"] = record["note_uid"]

    yield from deduped.values()


def iter_note_snippets(
    notes_df,
    trigger_regex,
    *,
    context_chars=_LONGITUDINAL_PROFILE.context_chars,
    snippet_max_chars=_LONGITUDINAL_PROFILE.max_chars,
    max_workers=1,
):
    """Yield one snippet dict per trigger-bearing note, deduplicated per patient.

    Each yielded dict has: note_uid, DFCI_MRN, note_date (ISO or None), note_type,
    trigger_categories (sorted list), snippet. Copy-forward notes with identical
    cleaned snippet text are collapsed per patient, keeping the EARLIEST note_date
    as the provenance/fallback date (so criterion-onset dates are not inflated).

    Thin wrapper over scan_note_candidates() + dedupe_candidates() kept for
    backward compatibility / single-call convenience; group_patient_snippets calls
    those two steps directly so it can report intermediate counts.
    """
    if notes_df.is_empty():
        return
    candidates = scan_note_candidates(
        notes_df,
        context_chars=context_chars,
        snippet_max_chars=snippet_max_chars,
        max_workers=max_workers,
        trigger_regex=trigger_regex,
    )
    yield from dedupe_candidates(candidates)


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
    context_chars=_LONGITUDINAL_PROFILE.context_chars,
    snippet_max_chars=_LONGITUDINAL_PROFILE.max_chars,
    payload_max_chars=_LONGITUDINAL_PROFILE.payload_max_chars,
    max_workers=None,
):
    """Group deduped per-note snippets by patient into payload-sized chunks.

    Returns {mrn: [chunk, ...]} where each chunk is a list of snippet records
    (same dicts iter_note_snippets yields). Snippets are de-duplicated per patient,
    ordered chronologically (oldest first), then packed greedily up to
    `payload_max_chars`. Chunking — rather than capping — means NO snippet is
    dropped, so earliest-occurrence dates and rare findings survive even for
    heavily-documented patients; the number of LLM calls is one per chunk
    (one for most patients, a few for outliers) instead of one per note.

    The per-note scan (clean + trigger match + snippet) runs in parallel across
    processes via scan_note_candidates (max_workers=None -> os.cpu_count());
    dedup and packing stay single-process because dedup must see the whole
    cohort (the same snippet can arrive from different notes in different
    workers) and packing is comparatively cheap.
    """
    candidates = scan_note_candidates(
        notes_df,
        context_chars=context_chars,
        snippet_max_chars=snippet_max_chars,
        max_workers=max_workers,
        trigger_regex=trigger_regex,
    )

    by_mrn = {}
    for rec in dedupe_candidates(candidates):
        by_mrn.setdefault(rec["DFCI_MRN"], []).append(rec)

    patient_chunks = {}
    for mrn, recs in by_mrn.items():
        # Tiebreaker on `snippet` after note_date: chunk_index is a stage-2 resume
        # key (build_nepc_timeline.py), so sort order must be stable regardless of
        # scan parallelism / worker count or dict/hash-order variation across runs.
        recs.sort(key=lambda r: (r["note_date"] or "9999-99-99", r["snippet"]))
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
