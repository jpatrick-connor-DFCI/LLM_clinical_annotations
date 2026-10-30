"""PSA measurements from LABS and notes, and the PSA rules built on them.

`merge_psa` reconciles one patient's LABS results with the gated note PSA
findings. LABS wins for a measurement both report:

  - LABS rows sharing a specimen id are one draw. Without ids, same-day rows
    with compatible values are one draw (the most informative kept); same-day
    incompatible values are a conflict.
  - A note value is a restatement of a LABS draw when the LABS value, rounded
    to the decimals the note wrote, equals it (same comparator class) and the
    draw falls within NOTE_LABS_MATCH_DAYS before the note's date interval. The
    closest such draw wins. A restatement never counts as a separate draw.
  - Unmatched note values are separate draws, even near another LABS draw.
    Note values restating one another within NOTE_DEDUP_DAYS are merged.
  - Year-precision note values are kept only for audit.
  - Conflicting measurements are kept, flagged, and never establish a
    threshold crossing.

Values are compared through bounds, so "<0.2" never meets ">=0.2", and an
"undetectable" result with no stated limit counts as below UNDETECTABLE_CEILING
(the clearance threshold) but never sets a numeric nadir or baseline.

Rules (project defaults; bump PSA_RULES_VERSION on any change):

  persistent_psa  the first decidable PSA 30-180 days after RP is >= 0.1, with
                  no earlier clearance
  psa_clearance   the first PSA after RP certainly below 0.1 (or undetectable)
  aua_bcr         after clearance, a PSA >= 0.2 confirmed by a later separate
                  draw >= 0.2 with no value below 0.2 between them
  phoenix_bcr     after primary RT, a PSA >= the preceding nadir + 2
  pcwg3_progression
                  per therapy episode: a baseline within 30 days before the
                  start (same-day values excluded); a rise of >= 25% and >= 2
                  ng/mL over the on-treatment nadir (the baseline if no
                  decline) at least 12 weeks after the start, confirmed >= 3
                  weeks later within the episode, with no value below the
                  threshold in between
"""

import re
from datetime import date, timedelta

from tasks.onset import date_interval

PSA_RULES_VERSION = "course-psa-rules-v1"

NOTE_LABS_MATCH_DAYS = 7
NOTE_DEDUP_DAYS = 7
CLEARANCE_THRESHOLD = 0.1
UNDETECTABLE_CEILING = CLEARANCE_THRESHOLD
PERSISTENCE_WINDOW_DAYS = (30, 180)
PERSISTENCE_THRESHOLD = 0.1
AUA_THRESHOLD = 0.2
PHOENIX_RISE = 2.0
PCWG3_BASELINE_DAYS = 30
PCWG3_MIN_ONSET_DAYS = 84
PCWG3_CONFIRM_DAYS = 21
PCWG3_RISE_FRACTION = 0.25
PCWG3_RISE_ABSOLUTE = 2.0

RULE_DEFAULTS = {
    "psa_rules_version": PSA_RULES_VERSION,
    "note_labs_match_days": NOTE_LABS_MATCH_DAYS,
    "note_dedup_days": NOTE_DEDUP_DAYS,
    "clearance_threshold": CLEARANCE_THRESHOLD,
    "undetectable_counts_as_clearance": True,
    "persistence_window_days": list(PERSISTENCE_WINDOW_DAYS),
    "persistence_threshold": PERSISTENCE_THRESHOLD,
    "aua_threshold": AUA_THRESHOLD,
    "aua_confirmation": "second_separate_draw_at_or_above_threshold",
    "phoenix_rise": PHOENIX_RISE,
    "pcwg3_baseline_days": PCWG3_BASELINE_DAYS,
    "pcwg3_same_day_baseline": "excluded",
    "pcwg3_min_onset_days": PCWG3_MIN_ONSET_DAYS,
    "pcwg3_confirm_days": PCWG3_CONFIRM_DAYS,
    "pcwg3_rise": [PCWG3_RISE_FRACTION, PCWG3_RISE_ABSOLUTE],
}

BELOW = {"<", "<=", "undetectable"}
ABOVE = {">", ">="}
_NUMBER = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)(?![\d.])")


def _d(value):
    return date.fromisoformat(value) if isinstance(value, str) else value


def comparator_class(comparator):
    if comparator in BELOW:
        return "below"
    if comparator in ABOVE:
        return "above"
    return "exact"


def upper(m):
    """Largest value the measurement allows; None when unbounded or unknown."""
    if m["comparator"] == "undetectable":
        return None
    if m["comparator"] in ABOVE:
        return None
    return m["value"]


def lower(m):
    if m["comparator"] in BELOW:
        return 0.0
    return m["value"]


def certainly_below(m, threshold):
    if m["comparator"] == "undetectable":
        return threshold >= UNDETECTABLE_CEILING
    if m["comparator"] == "=":
        return m["value"] < threshold
    if m["comparator"] == "<":
        return m["value"] <= threshold
    if m["comparator"] == "<=":
        return m["value"] < threshold
    return False


def certainly_at_least(m, threshold):
    if m["comparator"] in {"=", ">="}:
        return m["value"] >= threshold
    if m["comparator"] == ">":
        return m["value"] >= threshold
    return False


def decidable(m, threshold):
    return certainly_below(m, threshold) or certainly_at_least(m, threshold)


def _compatible(a, b):
    """Two results that could be the same draw's value."""
    if a["comparator"] == "=" and b["comparator"] == "=":
        return abs(a["value"] - b["value"]) < 1e-9
    lo = max(lower(a) or 0.0, lower(b) or 0.0)
    ups = [u for u in (upper(a), upper(b)) if u is not None]
    if a["comparator"] == "undetectable" or b["comparator"] == "undetectable":
        ups.append(UNDETECTABLE_CEILING)
    return not ups or lo <= min(ups)


INFORMATIVE = {"=": 3, "<": 2, "<=": 2, ">": 1, ">=": 1, "undetectable": 0}


def _absorb(keep, other):
    """Merge `other` into `keep` (same draw), keeping the more informative value."""
    if INFORMATIVE[other["comparator"]] > INFORMATIVE[keep["comparator"]] or (
        other["comparator"] in BELOW and keep["comparator"] in BELOW
        and (upper(other) or 1e9) < (upper(keep) or 1e9)
    ):
        for key in ("value", "comparator"):
            keep[key] = other[key]
    keep["sources"] = sorted(set(keep["sources"]) | set(other["sources"]))
    if other.get("quote") and not keep.get("quote"):
        keep["quote"] = other["quote"]


def _measurement(when, precision, value, comparator, source, source_id, quote=None,
                 specimen_id=None):
    lo, hi = date_interval(when.isoformat(), precision)
    return {
        "date": when, "precision": precision, "lo": lo, "hi": hi,
        "value": value, "comparator": comparator, "source": source,
        "source_id": source_id, "sources": [source_id], "specimen_id": specimen_id,
        "conflict": False, "quote": quote,
    }


def labs_measurements(rows):
    """LABS PSA rows (dicts from preprocessing.structured.load_psa) as measurements."""
    by_specimen, by_day = {}, {}
    out = []
    for row in sorted(rows, key=lambda r: (r["date"], r["source_id"])):
        if row["comparator"] != "undetectable" and row["psa_ng_ml"] is None:
            continue
        m = _measurement(_d(row["date"]), "day", row["psa_ng_ml"], row["comparator"],
                         "labs", row["source_id"], specimen_id=row.get("specimen_id"))
        same = by_specimen.get(m["specimen_id"]) if m["specimen_id"] else None
        if same is not None:
            if _compatible(same, m):
                _absorb(same, m)
            else:
                same["conflict"] = True
                same["sources"] = sorted(set(same["sources"]) | {m["source_id"]})
            continue
        target = None
        for other in by_day.get(m["date"], []):
            if m["specimen_id"] and other["specimen_id"] and m["specimen_id"] != other["specimen_id"]:
                continue
            if _compatible(other, m):
                _absorb(other, m)
                target = other
                break
            other["conflict"] = m["conflict"] = True
        if target is None:
            by_day.setdefault(m["date"], []).append(m)
            out.append(m)
            target = m
        if m["specimen_id"]:
            by_specimen[m["specimen_id"]] = target
    return out


def _decimals(value, quote):
    """Decimal places the note wrote for `value` (from the quote text)."""
    best = None
    for text in _NUMBER.findall(quote or ""):
        if abs(float(text) - value) < 1e-9:
            places = len(text.split(".")[1]) if "." in text else 0
            best = places if best is None else max(best, places)
    if best is None:
        text = repr(float(value))
        best = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return best


def restates(note, labs):
    """Whether a note value could be a rounded restatement of a LABS result."""
    cls = comparator_class(note["comparator"])
    if cls == "exact":
        if labs["comparator"] != "=":
            return False
        places = _decimals(note["value"], note.get("quote"))
        return abs(round(labs["value"], places) - note["value"]) < 1e-9
    if cls == "below":
        ceiling = UNDETECTABLE_CEILING if note["comparator"] == "undetectable" else note["value"]
        if labs["comparator"] in BELOW:
            if note["comparator"] == "undetectable" or labs["comparator"] == "undetectable":
                return True
            return abs(labs["value"] - note["value"]) < 1e-9
        return labs["comparator"] == "=" and labs["value"] < ceiling
    return certainly_at_least(labs, note["value"])


def note_measurements(rows):
    """(measurements, audit) from gated psa_value findings of one patient."""
    out, audit = [], []
    for row in rows:
        if row.get("item") != "psa_value" or not row.get("event_date"):
            continue
        comparator = row.get("value_comparator") or "="
        value = row.get("psa_ng_ml")
        source_id = f"note:{row['note_uid']}:{row['finding_index']}"
        if comparator != "undetectable" and value is None:
            audit.append({"source_id": source_id, "reason": "no_value"})
            continue
        if row.get("date_precision") not in {"day", "month"}:
            audit.append({"source_id": source_id, "reason": f"precision_{row.get('date_precision')}"})
            continue
        out.append(_measurement(_d(row["event_date"]), row["date_precision"], value,
                                comparator, "note", source_id, quote=row.get("quote")))
    return out, audit


def merge_psa(labs_rows, note_rows):
    """One patient's reconciled PSA measurements, ordered, plus an audit list."""
    labs = labs_measurements(labs_rows)
    notes, audit = note_measurements(note_rows)
    unmatched = []
    for note in notes:
        window = (note["lo"] - timedelta(days=NOTE_LABS_MATCH_DAYS), note["hi"])
        matches = [
            m for m in labs
            if window[0] <= m["date"] <= window[1] and not m["conflict"] and restates(note, m)
        ]
        if matches:
            best = min(matches, key=lambda m: ((window[1] - m["date"]).days, m["source_id"]))
            best["sources"] = sorted(set(best["sources"]) | {note["source_id"]})
            audit.append({"source_id": note["source_id"], "reason": f"restates:{best['source_id']}"})
            continue
        unmatched.append(note)

    merged_notes = []
    for note in sorted(unmatched, key=lambda m: (m["lo"], m["source_id"])):
        twin = next((
            other for other in merged_notes
            if comparator_class(other["comparator"]) == comparator_class(note["comparator"])
            and other["value"] == note["value"]
            and note["lo"] <= other["hi"] + timedelta(days=NOTE_DEDUP_DAYS)
            and other["lo"] <= note["hi"] + timedelta(days=NOTE_DEDUP_DAYS)
        ), None)
        if twin is None:
            merged_notes.append(note)
            continue
        if note["precision"] == "day" and twin["precision"] != "day":
            for key in ("date", "precision", "lo", "hi"):
                twin[key] = note[key]
        _absorb(twin, note)
        audit.append({"source_id": note["source_id"], "reason": f"restates:{twin['source_id']}"})

    measurements = labs + merged_notes
    day = {}
    for m in measurements:
        if m["precision"] == "day":
            day.setdefault(m["date"], []).append(m)
    for same_day in day.values():
        for i, a in enumerate(same_day):
            for b in same_day[i + 1:]:
                if a["source"] == "labs" and b["source"] == "labs":
                    continue
                if not _compatible(a, b):
                    a["conflict"] = b["conflict"] = True
    measurements.sort(key=lambda m: (m["lo"], m["hi"], m["source_id"]))
    return measurements, audit


def _interval(point):
    """(lo, hi) date bounds of an index event {"event_date", "date_precision"}."""
    if point is None or not point.get("event_date"):
        return None
    return date_interval(point["event_date"], point.get("date_precision") or "day")


def _usable(measurements):
    return [m for m in measurements if not m["conflict"]]


def summary(m):
    if m is None:
        return {"id": None, "date": None, "precision": None, "value": None, "comparator": None}
    return {
        "id": m["source_id"], "date": m["date"].isoformat(), "precision": m["precision"],
        "value": m["value"], "comparator": m["comparator"],
    }


def persistent_psa(measurements, rp):
    """{"status": True|False|None, "measurement": m|None, "reason"} after RP."""
    bounds = _interval(rp)
    if bounds is None:
        return {"status": None, "measurement": None, "reason": "no_rp_date"}
    rp_lo, rp_hi = bounds
    window_lo = rp_hi + timedelta(days=PERSISTENCE_WINDOW_DAYS[0])
    window_hi = rp_lo + timedelta(days=PERSISTENCE_WINDOW_DAYS[1])
    for m in _usable(measurements):
        if m["lo"] <= rp_hi:
            continue
        if m["lo"] < window_lo:
            if certainly_below(m, PERSISTENCE_THRESHOLD):
                return {"status": False, "measurement": m, "reason": "early_clearance"}
            continue
        if m["hi"] > window_hi:
            break
        if certainly_at_least(m, PERSISTENCE_THRESHOLD):
            return {"status": True, "measurement": m, "reason": None}
        if certainly_below(m, PERSISTENCE_THRESHOLD):
            return {"status": False, "measurement": m, "reason": None}
    return {"status": None, "measurement": None, "reason": "no_decidable_psa_in_window"}


def psa_clearance(measurements, rp):
    """The first post-RP measurement certainly below the clearance threshold."""
    bounds = _interval(rp)
    if bounds is None:
        return None
    for m in _usable(measurements):
        if m["lo"] > bounds[1] and certainly_below(m, CLEARANCE_THRESHOLD):
            return m
    return None


def _result(status, reason=None, event=None, confirmation=None, **extra):
    return {
        "status": status, "reason": reason,
        "date": event["date"].isoformat() if event else None,
        "precision": event["precision"] if event else None,
        "event": event, "confirmation": confirmation, **extra,
    }


def aua_bcr(measurements, rp):
    """AUA BCR after RP: requires documented clearance, then a confirmed rise."""
    bounds = _interval(rp)
    if bounds is None:
        return _result("unevaluable", "no_rp_date")
    after_rp = [m for m in measurements if m["lo"] > bounds[1]]
    if not after_rp:
        return _result("unevaluable", "no_post_rp_psa")
    clearance = psa_clearance(measurements, rp)
    if clearance is None:
        return _result("unevaluable", "no_clearance")
    later = [m for m in _usable(measurements) if m["lo"] > clearance["hi"]]
    pending = None
    for index, candidate in enumerate(later):
        if not certainly_at_least(candidate, AUA_THRESHOLD):
            continue
        for confirm in later[index + 1:]:
            if confirm["lo"] <= candidate["hi"]:
                continue
            if certainly_below(confirm, AUA_THRESHOLD):
                break
            if certainly_at_least(confirm, AUA_THRESHOLD):
                return _result("met", None, candidate, confirm, clearance=clearance)
        else:
            pending = pending or candidate
    if pending is not None:
        return _result("unevaluable", "unconfirmed", clearance=clearance, candidate=pending)
    if any(m["conflict"] and m["lo"] > clearance["hi"] for m in measurements):
        return _result("unevaluable", "conflicting_values", clearance=clearance)
    return _result("not_met", None, clearance=clearance)


def phoenix_bcr(measurements, rt):
    """Phoenix BCR after primary RT: PSA >= preceding nadir + 2."""
    bounds = _interval(rt)
    if bounds is None:
        return _result("unevaluable", "no_rt_date")
    nadir = None
    nadir_m = None
    seen = False
    for m in _usable(measurements):
        if m["lo"] <= bounds[1]:
            continue
        seen = True
        if nadir is not None and certainly_at_least(m, nadir + PHOENIX_RISE):
            return _result("met", None, m, None, nadir=nadir_m)
        bound = upper(m)
        if bound is not None and (nadir is None or bound < nadir):
            nadir, nadir_m = bound, m
    if not seen:
        return _result("unevaluable", "no_post_rt_psa")
    if nadir is None:
        return _result("unevaluable", "no_numeric_nadir")
    return _result("not_met", None, nadir=nadir_m)


def pcwg3_progression(measurements, start, end=None):
    """PCWG3 PSA progression within one therapy episode.

    `start` and `end` are {"event_date", "date_precision"}; `end` (the next
    episode's start) bounds the episode. Returns a _result with baseline,
    nadir, branch and threshold.
    """
    start_lo, start_hi = _interval(start)
    end_lo = _interval(end)[0] if end else None
    usable = _usable(measurements)
    baseline_lo = start_lo - timedelta(days=PCWG3_BASELINE_DAYS)
    candidates = [
        m for m in usable
        if m["hi"] < start_lo and m["lo"] >= baseline_lo
        and m["comparator"] in {"=", "<", "<="} and m["value"] is not None
    ]
    if not candidates:
        return _result("unevaluable", "no_baseline")
    baseline = max(candidates, key=lambda m: (m["hi"], m["lo"]))
    on_treatment = [
        m for m in usable
        if m["lo"] > start_hi and (end_lo is None or m["hi"] < end_lo)
    ]
    if not on_treatment:
        return _result("unevaluable", "no_on_treatment_psa", baseline=baseline)
    reference, reference_m = upper(baseline), baseline
    pending = None
    for index, m in enumerate(on_treatment):
        threshold = max(reference * (1 + PCWG3_RISE_FRACTION), reference + PCWG3_RISE_ABSOLUTE)
        eligible = (m["lo"] - start_hi).days >= PCWG3_MIN_ONSET_DAYS
        if eligible and certainly_at_least(m, threshold):
            for confirm in on_treatment[index + 1:]:
                if (confirm["lo"] - m["hi"]).days < PCWG3_CONFIRM_DAYS:
                    if certainly_below(confirm, threshold):
                        break
                    continue
                if certainly_below(confirm, threshold):
                    break
                if certainly_at_least(confirm, threshold):
                    return _result(
                        "met", None, m, confirm, baseline=baseline, nadir=reference_m,
                        branch="nadir" if reference_m is not baseline else "baseline",
                        threshold=threshold,
                    )
            else:
                pending = pending or m
        bound = upper(m)
        if bound is not None and bound < reference:
            reference, reference_m = bound, m
    if pending is not None:
        return _result("unevaluable", "unconfirmed", baseline=baseline, candidate=pending)
    return _result("not_met", None, baseline=baseline, nadir=reference_m)
