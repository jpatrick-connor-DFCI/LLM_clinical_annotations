"""Per-note scanning and per-patient ranking/packing of trigger snippets."""

import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from preprocessing.config import SNIPPET_PROFILES
from preprocessing.notes import to_iso_date
from preprocessing.parquet_io import write_parquet_atomic
from preprocessing.triggers import TRIGGER_REGEX, build_snippet, find_trigger_matches
from preprocessing.utils import clean_note

_BINARY_NEPC_PROFILE = SNIPPET_PROFILES["binary_nepc"]


def _scan_note_row(row, *, context_chars, snippet_max_chars, trigger_regex=TRIGGER_REGEX):
    """Clean, trigger-scan, and snippet a single note row.

    Returns a candidate dict (mrn + snippet metadata) or None if the note has no
    usable text or no trigger hit. Module-level so it can be pickled by a
    ProcessPoolExecutor worker.
    """
    note_text = row.get("CLINICAL_TEXT") or ""
    note_type = row.get("NOTE_TYPE") or "Unknown"
    cleaned = clean_note(note_text, note_type=note_type)
    if not cleaned:
        return None
    matches = find_trigger_matches(cleaned, trigger_regex)
    if not matches:
        return None
    snippet = build_snippet(
        cleaned,
        matches,
        context_chars=context_chars,
        max_chars=snippet_max_chars,
    )
    if not snippet:
        return None
    return {
        "mrn": int(row["DFCI_MRN"]),
        "note_date": to_iso_date(row.get("EVENT_DATE")),
        "note_type": note_type,
        "trigger_categories": sorted({m[0] for m in matches}),
        "trigger_count": len(matches),
        "snippet": snippet,
        "raw_note_id": row.get("RAW_NOTE_ID"),
    }


def _scan_note_chunk(rows, *, context_chars, snippet_max_chars, trigger_regex=TRIGGER_REGEX):
    """Scan a list of note rows in one worker call (amortizes IPC overhead)."""
    out = []
    for row in rows:
        candidate = _scan_note_row(
            row,
            context_chars=context_chars,
            snippet_max_chars=snippet_max_chars,
            trigger_regex=trigger_regex,
        )
        if candidate is not None:
            out.append(candidate)
    return out


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def scan_note_candidates(
    notes_df,
    *,
    context_chars=_BINARY_NEPC_PROFILE.context_chars,
    snippet_max_chars=_BINARY_NEPC_PROFILE.max_chars,
    max_workers=None,
    trigger_regex=TRIGGER_REGEX,
    progress_desc="Scanning notes",
):
    """Clean + trigger-scan + snippet every note, in parallel across processes.

    Returns {mrn: [candidate, ...]}. This is the CPU-heavy startup step; it is
    embarrassingly parallel across notes, so it is farmed out to a process pool.
    A worker count of 1 runs inline (useful for small cohorts / debugging).

    `trigger_regex` is a plain {label: pattern_str} dict (not compiled `re`
    objects), so it pickles cleanly across the ProcessPoolExecutor boundary.
    """
    rows = list(notes_df.iter_rows(named=True))
    if not rows:
        return {}

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, len(rows)))

    candidates = {}

    def _collect(results):
        for c in results:
            candidates.setdefault(c["mrn"], []).append(c)

    progress = tqdm(total=len(rows), desc=progress_desc, unit="note", dynamic_ncols=True)
    if max_workers == 1:
        chunk_size = min(256, len(rows))
        try:
            for chunk in _chunked(rows, chunk_size):
                _collect(
                    _scan_note_chunk(
                        chunk,
                        context_chars=context_chars,
                        snippet_max_chars=snippet_max_chars,
                        trigger_regex=trigger_regex,
                    )
                )
                progress.update(len(chunk))
        finally:
            progress.close()
        return candidates

    # ~4 chunks per worker keeps the pool fed while amortizing per-task pickling.
    chunk_size = max(1, math.ceil(len(rows) / (max_workers * 4)))
    chunks = list(_chunked(rows, chunk_size))
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            result_batches = executor.map(
                partial(
                    _scan_note_chunk,
                    context_chars=context_chars,
                    snippet_max_chars=snippet_max_chars,
                    trigger_regex=trigger_regex,
                ),
                chunks,
            )
            for chunk, results in zip(chunks, result_batches):
                _collect(results)
                progress.update(len(chunk))
    finally:
        progress.close()
    return candidates


def rank_patient_candidates(
    candidates,
    *,
    max_notes_per_patient,
    payload_max_chars,
    all_mrns=None,
):
    """Rank each patient's candidate notes and keep the top slice under budget.

    Ranking: (number of trigger categories, raw trigger count, recency), descending.
    Kept until either `max_notes_per_patient` or the cumulative `payload_max_chars`
    budget is hit, so outlier patients can't exceed the model's context window.
    When `all_mrns` is supplied, progress covers the full cohort, including
    patients without notes or trigger candidates.
    """
    ranked = {}
    mrns = sorted(
        {int(mrn) for mrn in all_mrns}
        if all_mrns is not None
        else candidates
    )
    for mrn in tqdm(
        mrns,
        total=len(mrns),
        desc="Compiling binary NEPC patients",
        unit="patient",
        dynamic_ncols=True,
    ):
        items = candidates.get(mrn, [])
        if not items:
            continue
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


def _snippet_cache_key(notes_df, *, max_notes_per_patient, snippet_max_chars, payload_max_chars, context_chars):
    """Deterministic key over the note content + all params that affect the output.

    Hashes the (mrn, date, type, text) of every note plus the snippet-building
    params and the trigger-regex source, so any change to inputs or logic misses
    the cache rather than returning a stale result.
    """
    hasher = hashlib.sha256()
    for name in ("DFCI_MRN", "EVENT_DATE", "NOTE_TYPE", "CLINICAL_TEXT"):
        if name in notes_df.columns:
            col = notes_df.get_column(name)
            hasher.update(name.encode())
            hasher.update(str(col.hash().sum()).encode())
    for label, pattern in TRIGGER_REGEX.items():
        hasher.update(label.encode())
        hasher.update(pattern.encode())
    hasher.update(
        repr((
            int(notes_df.height),
            int(context_chars),
            int(max_notes_per_patient),
            int(snippet_max_chars),
            int(payload_max_chars),
        )).encode()
    )
    return hasher.hexdigest()[:16]


def build_patient_snippets(
    notes_df,
    *,
    all_mrns=None,
    max_notes_per_patient=75,
    context_chars=_BINARY_NEPC_PROFILE.context_chars,
    snippet_max_chars=_BINARY_NEPC_PROFILE.max_chars,
    payload_max_chars=_BINARY_NEPC_PROFILE.payload_max_chars,
    max_workers=None,
    cache_dir=None,
):
    """Return {mrn: [{note_date, note_type, trigger_categories, snippet}, ...]}.

    Notes without any trigger hit are dropped. The per-note scan (clean + trigger
    match + snippet) runs in parallel across processes. Results are ranked per
    patient and capped by `max_notes_per_patient` / `payload_max_chars`. Pass
    `all_mrns` to report patient-level progress over the complete cohort.

    Sizing defaults match `SNIPPET_PROFILES["binary_nepc"]`; callers for other
    tasks should pass values from their own `SnippetProfile` explicitly.

    If `cache_dir` is given, the ranked result is cached under a content+params
    hash so re-runs over the same cohort skip the whole scan.
    """
    if notes_df.is_empty():
        if all_mrns is not None:
            for _ in tqdm(
                sorted({int(mrn) for mrn in all_mrns}),
                desc="Compiling binary NEPC patients",
                unit="patient",
                dynamic_ncols=True,
            ):
                pass
        return {}

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        key = _snippet_cache_key(
            notes_df,
            max_notes_per_patient=max_notes_per_patient,
            snippet_max_chars=snippet_max_chars,
            payload_max_chars=payload_max_chars,
            context_chars=context_chars,
        )
        cache_path = cache_dir / f"patient_snippets_{key}.parquet"
        if cache_path.exists():
            try:
                cached = pl.read_parquet(cache_path).sort(["DFCI_MRN", "snippet_index"])
                grouped = {}
                for row in cached.iter_rows(named=True):
                    grouped.setdefault(int(row["DFCI_MRN"]), []).append(
                        {
                            "note_date": row["note_date"],
                            "note_type": row["note_type"],
                            "trigger_categories": row["trigger_categories"] or [],
                            "snippet": row["snippet"],
                        }
                    )
                return grouped
            except (OSError, pl.exceptions.PolarsError, ValueError):
                pass  # corrupt/partial cache — fall through and recompute

    candidates = scan_note_candidates(
        notes_df,
        context_chars=context_chars,
        snippet_max_chars=snippet_max_chars,
        max_workers=max_workers,
    )
    ranked = rank_patient_candidates(
        candidates,
        max_notes_per_patient=max_notes_per_patient,
        payload_max_chars=payload_max_chars,
        all_mrns=all_mrns,
    )

    if cache_path is not None:
        rows = []
        for mrn, snippets in ranked.items():
            for snippet_index, snippet in enumerate(snippets):
                rows.append(
                    {
                        "DFCI_MRN": int(mrn),
                        "snippet_index": snippet_index,
                        **snippet,
                    }
                )
        if rows:
            write_parquet_atomic(pl.DataFrame(rows, strict=False), cache_path)

    return ranked
