"""Therapy exposure from structured medications and gated note statements.

Exposure is an interval with a certainty tier, never a blanket window:

  supported   structured records establish it: the inter-dose interval of
              repeated records of one agent, a parseable formulation, days
              supply or stop date, an agent with a fixed schedule (degarelix
              monthly), or an orchiectomy (permanent).
  reported    a dated note statement: "continues on Lupron" holds at its
              assessment date; a stated start followed by a current statement,
              with no stop between, holds across the span between them.
  estimated   a record whose duration nothing establishes (a single depot
              injection of unknown formulation, an order without supply). It
              is recorded as possible exposure and its assumption is named,
              but it never supports castration or CRPC.

Repeated records of one agent: each record is covered until the next one, and
the last record of a run for that patient's median observed interval (capped
at DEPOT_MAX_COVERAGE_DAYS). A gap longer than GAP_FACTOR x the interval ends
supported coverage; the remainder of a gap up to ESTIMATED_SPLIT_DAYS is
estimated, and a longer gap splits the episode.

Therapy episodes for PCWG3 start at a new agent, or at a restart after a gap
longer than ESTIMATED_SPLIT_DAYS; refills and repeated depot doses do not
start an episode.

Bump THERAPY_VERSION on any change; builder metadata records it.
"""

import re
from datetime import date, timedelta
from statistics import median

from tasks.onset import date_interval

THERAPY_VERSION = "course-therapy-v1"

GAP_FACTOR = 1.5
DEPOT_MAX_COVERAGE_DAYS = 180
ESTIMATED_SPLIT_DAYS = 270
ESTIMATED_DEFAULT_DAYS = 90
ESTIMATED_DEPOT_DAYS = 180
REPORTED_LINK_DAYS = 180
REPORTED_POINT_TOLERANCE_DAYS = 30
EPISODE_MERGE_DAYS = 60
MIN_REPEAT_INTERVAL_DAYS = 14

CASTRATING_CLASSES = frozenset({"depot_adt", "oral_adt", "adt_unspecified", "orchiectomy"})
# Classes that start a PCWG3 therapy episode (the backbone ADT included).
EPISODE_CLASSES = frozenset({
    "depot_adt", "oral_adt", "adt_unspecified", "orchiectomy", "antiandrogen",
    "arpi", "taxane", "radium", "lu_psma", "sipuleucel", "parpi", "platinum",
})

# Note agent text -> (drug_class, agent). Order matters: the first match wins.
NOTE_AGENTS = [
    (r"leuprolide|lupron|eligard|camcevi|fensolvi", "depot_adt", "leuprolide"),
    (r"goserelin|zoladex", "depot_adt", "goserelin"),
    (r"triptorelin|trelstar", "depot_adt", "triptorelin"),
    (r"histrelin|vantas|supprelin", "depot_adt", "histrelin"),
    (r"degarelix|firmagon", "depot_adt", "degarelix"),
    (r"relugolix|orgovyx", "oral_adt", "relugolix"),
    (r"orchiectomy|orchidectomy", "orchiectomy", "orchiectomy"),
    (r"bicalutamide|casodex", "antiandrogen", "bicalutamide"),
    (r"flutamide|eulexin", "antiandrogen", "flutamide"),
    (r"nilutamide|nilandron", "antiandrogen", "nilutamide"),
    (r"abiraterone|zytiga|yonsa|akeega", "arpi", "abiraterone"),
    (r"enzalutamide|xtandi", "arpi", "enzalutamide"),
    (r"apalutamide|erleada", "arpi", "apalutamide"),
    (r"darolutamide|nubeqa", "arpi", "darolutamide"),
    (r"cabazitaxel|jevtana", "taxane", "cabazitaxel"),
    (r"docetaxel|taxotere", "taxane", "docetaxel"),
    (r"radium|xofigo|ra[\s-]*223", "radium", "radium"),
    (r"lutetium|lu[\s-]*177|177[\s-]*lu|pluvicto|vipivotide|psma[\s-]*617",
     "lu_psma", "lutetium_psma"),
    (r"sipuleucel|provenge", "sipuleucel", "sipuleucel"),
    (r"olaparib|lynparza", "parpi", "olaparib"),
    (r"rucaparib|rubraca", "parpi", "rucaparib"),
    (r"niraparib|zejula", "parpi", "niraparib"),
    (r"talazoparib|talzenna", "parpi", "talazoparib"),
    (r"carboplatin", "platinum", "carboplatin"),
    (r"cisplatin", "platinum", "cisplatin"),
    (r"\badt\b|androgen[\s-]+deprivation|\b(?:lhrh|gnrh)\b|hormon(?:e|al)\s+therapy",
     "adt_unspecified", "adt"),
]
_NOTE_AGENTS = [(re.compile(p, re.IGNORECASE), c, a) for p, c, a in NOTE_AGENTS]

# Formulation -> days, by agent and dose (mg), from the FDA labels.
DOSE_DAYS = {
    "leuprolide": {7.5: 28, 22.5: 84, 30.0: 112, 45.0: 168, 3.75: 28, 11.25: 84},
    "goserelin": {3.6: 28, 10.8: 84},
    "triptorelin": {3.75: 28, 11.25: 84, 22.5: 168},
    "histrelin": {50.0: 365},
}
AGENT_SCHEDULE_DAYS = {"degarelix": 28, "histrelin": 365}
_MONTHS = re.compile(r"\b(\d{1,2})[\s-]*(?:month|mo)\b", re.IGNORECASE)
_MG = re.compile(r"(\d+(?:\.\d+)?)\s*mg\b", re.IGNORECASE)


def classify_agent(text):
    """(drug_class, agent) for a note's agent text, or None."""
    if not text:
        return None
    for pattern, drug_class, agent in _NOTE_AGENTS:
        if pattern.search(str(text)):
            return drug_class, agent
    return None


def _d(value):
    return date.fromisoformat(value) if isinstance(value, str) else value


def formulation_days(record):
    """Days one structured record covers, when its own fields establish it."""
    agent = record.get("agent")
    supply = record.get("days_supply")
    try:
        supply = float(supply) if supply not in (None, "") else None
    except (TypeError, ValueError):
        supply = None
    if supply and supply > 0:
        return int(supply)
    if agent in AGENT_SCHEDULE_DAYS:
        return AGENT_SCHEDULE_DAYS[agent]
    text = " ".join(
        str(record.get(k) or "") for k in ("formulation", "med_name", "frequency")
    )
    months = _MONTHS.search(text)
    if months and 1 <= int(months.group(1)) <= 12:
        return int(months.group(1)) * 28
    doses = DOSE_DAYS.get(agent, {})
    dose_text = f"{record.get('dose') or ''} {record.get('dose_unit') or ''} {text}"
    for mg in _MG.findall(dose_text):
        if float(mg) in doses:
            return doses[float(mg)]
    try:
        dose = float(record.get("dose")) if record.get("dose") not in (None, "") else None
    except (TypeError, ValueError):
        dose = None
    if dose in doses:
        return doses[dose]
    return None


def _interval(agent, drug_class, start, end, certainty, basis, source_ids):
    return {
        "agent": agent, "drug_class": drug_class, "start": start, "end": end,
        "certainty": certainty, "basis": basis, "source_ids": list(source_ids),
    }


def structured_intervals(records):
    """Exposure intervals from structured medication records of one patient."""
    by_agent = {}
    for record in records:
        when = record.get("administration_date") or record.get("date")
        if when:
            by_agent.setdefault(record["agent"], []).append({**record, "_date": _d(when)})
    intervals = []
    for agent, rows in by_agent.items():
        rows.sort(key=lambda r: r["_date"])
        drug_class = rows[0]["drug_class"]
        dates = sorted({r["_date"] for r in rows})
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        usable = [g for g in gaps if MIN_REPEAT_INTERVAL_DAYS <= g <= DEPOT_MAX_COVERAGE_DAYS]
        typical = min(int(median(usable)), DEPOT_MAX_COVERAGE_DAYS) if usable else None
        by_date = {}
        for row in rows:
            by_date.setdefault(row["_date"], []).append(row)
        for index, when in enumerate(dates):
            group = by_date[when]
            ids = [r["source_id"] for r in group]
            stop = max((_d(r["stop_date"]) for r in group if r.get("stop_date")), default=None)
            own = max((d for d in (formulation_days(r) for r in group) if d), default=None)
            nxt = dates[index + 1] if index + 1 < len(dates) else None
            if stop is not None and stop >= when:
                intervals.append(_interval(agent, drug_class, when, stop, "supported", "stop_date", ids))
                continue
            if own is not None:
                end = when + timedelta(days=own)
                intervals.append(_interval(agent, drug_class, when, end, "supported", "formulation", ids))
                if nxt is not None and end < nxt and (nxt - end).days <= ESTIMATED_SPLIT_DAYS:
                    intervals.append(_interval(agent, drug_class, end, nxt, "estimated", "gap_after_formulation", ids))
                continue
            if typical is not None:
                gap = (nxt - when).days if nxt else None
                if gap is not None and gap <= GAP_FACTOR * typical:
                    intervals.append(_interval(agent, drug_class, when, nxt, "supported", "dose_interval", ids))
                    continue
                end = when + timedelta(days=typical)
                intervals.append(_interval(agent, drug_class, when, end, "supported", "dose_interval_last", ids))
                if nxt is not None and (nxt - when).days <= ESTIMATED_SPLIT_DAYS:
                    intervals.append(_interval(agent, drug_class, end, nxt, "estimated", "dose_interval_gap", ids))
                continue
            span = ESTIMATED_DEPOT_DAYS if drug_class == "depot_adt" else ESTIMATED_DEFAULT_DAYS
            end = when + timedelta(days=span)
            if nxt is not None and nxt < end:
                end = nxt
            intervals.append(_interval(agent, drug_class, when, end, "estimated", "single_record", ids))
    return intervals


def _note_point(row):
    """(date, precision) a note finding speaks for; None when too coarse."""
    if row.get("date_precision") not in {"day", "month"} or not row.get("event_date"):
        return None
    return _d(row["event_date"]), row["date_precision"]


def note_intervals(rows):
    """Reported exposure from gated therapy_start / _current / _stop findings."""
    events = {}
    for row in rows:
        item = row.get("item")
        if item not in {"therapy_start", "therapy_current", "therapy_stop"}:
            continue
        match = classify_agent(row.get("agent")) or classify_agent(row.get("quote"))
        point = _note_point(row)
        if match is None or point is None:
            continue
        if item == "therapy_start" and row.get("date_source") != "stated":
            continue
        source = f"note:{row['note_uid']}:{row['finding_index']}"
        events.setdefault(match, []).append((point[0], item, source, point[1]))
    intervals = []
    for (drug_class, agent), items in events.items():
        items.sort()
        if drug_class == "orchiectomy":
            first = items[0]
            intervals.append(_interval(agent, drug_class, first[0], date.max, "supported",
                                       "orchiectomy", [first[2]]))
            continue
        open_from = None
        for when, item, source, _ in items:
            if item == "therapy_stop":
                if open_from is not None and when > open_from[0]:
                    intervals.append(_interval(agent, drug_class, open_from[0], when, "reported",
                                               "note_span", [open_from[1], source]))
                open_from = None
                continue
            intervals.append(_interval(agent, drug_class, when, when, "reported",
                                       f"note_{item.split('_')[1]}", [source]))
            if open_from is not None and 0 < (when - open_from[0]).days <= REPORTED_LINK_DAYS:
                intervals.append(_interval(agent, drug_class, open_from[0], when, "reported",
                                           "note_span", [open_from[1], source]))
            open_from = (when, source)
    return intervals


def stop_points(rows):
    """{(drug_class, agent): [stop dates]} from gated therapy_stop findings."""
    stops = {}
    for row in rows:
        if row.get("item") != "therapy_stop":
            continue
        match = classify_agent(row.get("agent")) or classify_agent(row.get("quote"))
        point = _note_point(row)
        if match and point:
            stops.setdefault(match, []).append(point[0])
    return stops


def patient_intervals(med_records, course_rows):
    """Every exposure interval for one patient (structured + reported)."""
    return structured_intervals(med_records) + note_intervals(course_rows)


def _covers(interval, when):
    start, end = interval["start"], interval["end"]
    if interval["certainty"] == "reported" and start == end:
        return abs((when - start).days) <= REPORTED_POINT_TOLERANCE_DAYS
    return start <= when <= end


TIER_RANK = {"supported": 3, "reported": 2, "estimated": 1}


def exposure_at(intervals, when, stops=None):
    """Therapy status on one date.

    Returns therapies_active (supported or reported), therapies_possible
    (estimated only), adt_active (True with supported/reported castration;
    False after an explicit ADT stop with nothing covering the date; else
    None), and the strongest ADT tier with its basis.
    """
    when = _d(when)
    tiers = {}
    for interval in intervals:
        if not _covers(interval, when):
            continue
        key = interval["agent"]
        best = tiers.get(key)
        if best is None or TIER_RANK[interval["certainty"]] > TIER_RANK[best["certainty"]]:
            tiers[key] = interval
    active = sorted(a for a, i in tiers.items() if i["certainty"] != "estimated")
    possible = sorted(a for a, i in tiers.items() if i["certainty"] == "estimated")
    adt = [i for i in tiers.values() if i["drug_class"] in CASTRATING_CLASSES]
    adt_best = max(adt, key=lambda i: TIER_RANK[i["certainty"]], default=None)
    if adt_best is not None and adt_best["certainty"] != "estimated":
        adt_active = True
    elif adt_best is None and stops and any(
        c in CASTRATING_CLASSES and any(s <= when for s in dates)
        for (c, _), dates in stops.items()
    ):
        adt_active = False
    else:
        adt_active = None
    return {
        "therapies_active": active,
        "therapies_possible": possible,
        "adt_active": adt_active,
        "adt_certainty": adt_best and adt_best["certainty"],
        "adt_basis": adt_best and adt_best["basis"],
        "adt_source_ids": adt_best["source_ids"] if adt_best else [],
    }


def episode_starts(med_records, course_rows):
    """Therapy episode starts for PCWG3: [{agent, drug_class, start, precision, source}].

    A structured run starts at its first record and restarts after a gap over
    ESTIMATED_SPLIT_DAYS. A stated note start of the same agent within
    EPISODE_MERGE_DAYS is the same episode; the earlier certain date wins.
    """
    starts = []
    by_agent = {}
    for record in med_records:
        when = record.get("administration_date") or record.get("date")
        if when and record["drug_class"] in EPISODE_CLASSES:
            by_agent.setdefault((record["drug_class"], record["agent"]), []).append(
                (_d(when), record["source_id"])
            )
    for (drug_class, agent), rows in by_agent.items():
        rows.sort()
        previous = None
        for when, source in rows:
            if previous is None or (when - previous).days > ESTIMATED_SPLIT_DAYS:
                starts.append({"agent": agent, "drug_class": drug_class, "start": when,
                               "precision": "day", "source": source})
            previous = when
    for row in course_rows:
        if row.get("item") != "therapy_start" or row.get("date_source") != "stated":
            continue
        match = classify_agent(row.get("agent")) or classify_agent(row.get("quote"))
        point = _note_point(row)
        if match is None or point is None or match[0] not in EPISODE_CLASSES:
            continue
        note_start = {
            "agent": match[1], "drug_class": match[0], "start": point[0],
            "precision": point[1], "source": f"note:{row['note_uid']}:{row['finding_index']}",
        }
        same = [
            s for s in starts
            if s["agent"] == note_start["agent"]
            and abs((s["start"] - note_start["start"]).days) <= EPISODE_MERGE_DAYS
        ]
        if not same:
            starts.append(note_start)
            continue
        existing = same[0]
        note_end = date_interval(note_start["start"].isoformat(), note_start["precision"])[1]
        if note_end < existing["start"]:
            existing.update({k: note_start[k] for k in ("start", "precision", "source")})
    starts.sort(key=lambda s: (s["start"], s["agent"]))
    return starts
