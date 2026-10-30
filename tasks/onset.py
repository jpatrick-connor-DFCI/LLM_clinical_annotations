"""Choosing the earliest dated finding when dates carry different precisions.

Same rule as `tasks/longitudinal_subtype/build_nepc_timeline._prefer_onset`,
which stays in place so that pipeline is unchanged. Each finding is a dict with
`event_date` (ISO), `date_precision` (year | month | day | unknown), and
`source_note_date`.
"""

import calendar
from datetime import date

PRECISION_RANK = {"unknown": 0, "year": 1, "month": 2, "day": 3}


def date_interval(event_date, precision):
    """The calendar interval an event date at `precision` could fall in."""
    if not event_date:
        return None
    parsed = date.fromisoformat(event_date)
    if precision == "year":
        return date(parsed.year, 1, 1), date(parsed.year, 12, 31)
    if precision == "month":
        last = calendar.monthrange(parsed.year, parsed.month)[1]
        return date(parsed.year, parsed.month, 1), date(parsed.year, parsed.month, last)
    return parsed, parsed


def prefer_onset(candidate, existing):
    """Prefer certainly earlier dates; within overlapping periods prefer precision."""
    if existing is None:
        return True
    candidate_interval = date_interval(candidate["event_date"], candidate["date_precision"])
    existing_interval = date_interval(existing["event_date"], existing["date_precision"])
    if candidate_interval is None:
        return False
    if existing_interval is None:
        return True
    if candidate_interval[1] < existing_interval[0]:
        return True
    if existing_interval[1] < candidate_interval[0]:
        return False
    candidate_rank = PRECISION_RANK.get(candidate["date_precision"], 0)
    existing_rank = PRECISION_RANK.get(existing["date_precision"], 0)
    if candidate_rank != existing_rank:
        return candidate_rank > existing_rank
    return (candidate["source_note_date"] or "9999-99-99") < (
        existing["source_note_date"] or "9999-99-99"
    )


def earliest(findings):
    """Return the earliest finding under `prefer_onset`, or None."""
    best = None
    for finding in findings:
        if prefer_onset(finding, best):
            best = finding
    return best


def both_hold(rows):
    """The first date at which every component finding held, or None.

    The date is the latest component date. Its precision is the coarsest among
    components not certainly earlier than that date, since any of them could
    still be the one that completes the combination.
    """
    if not rows or any(r is None or not r.get("event_date") for r in rows):
        return None
    latest = max(rows, key=lambda r: (r["event_date"], r.get("source_note_date") or ""))
    start = date.fromisoformat(latest["event_date"])
    open_rows = [
        r for r in rows
        if date_interval(r["event_date"], r.get("date_precision"))[1] >= start
    ]
    precision = min(
        (r.get("date_precision") or "unknown" for r in open_rows),
        key=lambda p: PRECISION_RANK.get(p, 0),
    )
    return {
        "event_date": latest["event_date"],
        "date_precision": precision,
        "source_note_date": max((r.get("source_note_date") or "") for r in rows) or None,
    }
