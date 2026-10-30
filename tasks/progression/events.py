"""Dated progression events for one patient, each with the therapy it occurred on.

Sources:
  statement  gated progression_event findings (psa, radiographic, clinical,
             unspecified). Imaging recurrence findings are NOT promoted to
             progression: a radiographic event needs an explicit new or
             worsening finding or a progression assessment.
  pcwg3      PSA progression per therapy episode (tasks/disease_course/psa.py),
             with baseline and nadir reset at each new agent or restart.
  registry   death from the vital-status registry.

Statement events of one type on one date (and precision) are one event; its
sources are all kept. Every event carries the therapy exposure on its date
(tasks/disease_course/therapy.exposure_at). Unknown exposure stays unknown.

PFS itself is chosen downstream: `first_event_after` returns each patient's
first event of the requested types after an index date.

Bump PROGRESSION_EVENTS_VERSION on any change; metadata records it.
"""

from datetime import date

from tasks.disease_course.psa import merge_psa, pcwg3_progression, summary
from tasks.disease_course.therapy import (
    episode_starts,
    exposure_at,
    patient_intervals,
    stop_points,
)
from tasks.onset import date_interval

PROGRESSION_EVENTS_VERSION = "progression-events-v1"
STATEMENT_TYPES = ("psa", "radiographic", "clinical", "unspecified")
EVENT_TYPES = STATEMENT_TYPES + ("death",)


def _source(row):
    return f"note:{row['note_uid']}:{row['finding_index']}"


def _event(event_type, basis, when, precision, sources, **extra):
    return {
        "event_type": event_type, "basis": basis, "event_date": when,
        "date_precision": precision, "source_ids": sorted(sources), **extra,
    }


def statement_events(rows):
    merged = {}
    for row in rows:
        if row["item"] != "progression_event":
            continue
        event_type = row.get("progression_type") or "unspecified"
        key = (event_type, row["event_date"], row["date_precision"])
        event = merged.get(key)
        if event is None:
            merged[key] = _event(
                event_type, "statement", row["event_date"], row["date_precision"],
                [_source(row)], quote=row["quote"], therapy_stated=row.get("therapy_stated"),
            )
        else:
            event["source_ids"] = sorted(set(event["source_ids"]) | {_source(row)})
            event["therapy_stated"] = event["therapy_stated"] or row.get("therapy_stated")
    return list(merged.values())


def pcwg3_events(measurements, episodes):
    """(events, unevaluable reasons) from PCWG3 per therapy episode."""
    events, reasons = [], []
    for index, episode in enumerate(episodes):
        start = {"event_date": episode["start"].isoformat(), "date_precision": episode["precision"]}
        nxt = episodes[index + 1] if index + 1 < len(episodes) else None
        end = nxt and {"event_date": nxt["start"].isoformat(), "date_precision": nxt["precision"]}
        result = pcwg3_progression(measurements, start, end)
        label = f"{episode['agent']}@{start['event_date']}"
        if result["status"] != "met":
            if result["status"] == "unevaluable":
                reasons.append(f"{label}:{result['reason']}")
            continue
        candidate, confirmation = result["event"], result["confirmation"]
        events.append(_event(
            "psa", "pcwg3", result["date"], result["precision"],
            set(candidate["sources"]) | set(confirmation["sources"]),
            confirmation_date=confirmation["date"].isoformat(),
            episode_agent=episode["agent"], episode_start=start["event_date"],
            episode_source=episode["source"], pcwg3_branch=result["branch"],
            baseline_id=summary(result["baseline"])["id"],
            nadir_id=summary(result["nadir"])["id"],
            psa_value=candidate["value"], psa_threshold=result["threshold"],
        ))
    return events, reasons


def death_event(vital):
    if not vital or not vital.get("death_date"):
        return []
    return [_event("death", "registry", vital["death_date"], "day",
                   vital.get("registry_source_ids") or [])]


def with_exposure(event, intervals, stops):
    when = date_interval(event["event_date"], event["date_precision"])[0]
    exposure = exposure_at(intervals, when, stops)
    return {
        **event,
        "therapies_active": exposure["therapies_active"],
        "therapies_possible": exposure["therapies_possible"],
        "adt_active": exposure["adt_active"],
        "exposure_basis": exposure["adt_basis"],
        "exposure_certainty": exposure["adt_certainty"],
    }


def patient_events(rows, psa_rows, med_rows, vital, notes_reviewed=True):
    """(events, pcwg3 unevaluable reasons) for one patient.

    With `notes_reviewed` False (the course review failed or never ran) only
    the registry death is reported: PCWG3 without the note PSAs and therapy
    starts could miss an intervening value or an episode boundary.
    """
    events = death_event(vital)
    reasons = []
    if notes_reviewed:
        measurements, _ = merge_psa(psa_rows, rows)
        episodes = episode_starts(med_rows, rows)
        derived, reasons = pcwg3_events(measurements, episodes)
        events += statement_events(rows) + derived
    intervals = patient_intervals(med_rows, rows)
    stops = stop_points(rows)
    events = [with_exposure(e, intervals, stops) for e in events]
    events.sort(key=lambda e: (e["event_date"], e["event_type"], e["basis"]))
    return events, reasons


def first_event_after(events, index_dates, types=STATEMENT_TYPES):
    """{mrn: first event row of `types` certainly after that patient's index date}.

    `events` is an iterable of event dicts (or a polars frame's named rows)
    with DFCI_MRN; `index_dates` maps MRN to an ISO date.
    """
    if hasattr(events, "iter_rows"):
        events = events.iter_rows(named=True)
    first = {}
    for event in events:
        mrn = int(event["DFCI_MRN"])
        index = index_dates.get(mrn)
        if index is None or event["event_type"] not in types:
            continue
        lo, _ = date_interval(event["event_date"], event["date_precision"])
        if lo <= date.fromisoformat(index):
            continue
        best = first.get(mrn)
        if best is None or lo < date_interval(best["event_date"], best["date_precision"])[0]:
            first[mrn] = event
    return first
