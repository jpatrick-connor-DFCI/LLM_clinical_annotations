"""mCRPC diagnosis date for one patient.

CRPC holds from the earliest of:
  statement  a gated crpc_statement or mcrpc_statement
  pcwg3      PCWG3 PSA progression in a therapy episode while castrate:
               measured  testosterone < 50 ng/dL within 60 days of the rise
               assumed   no testosterone in that window, and supported or
                         reported ADT at both the rise and the confirmation
             Testosterone >= 50 in the window vetoes the episode. ADT known
             only from an estimated medication tail leaves it unevaluable.

Metastatic disease comes from an upstream label (AVPC by default, or
met_diagnosis), with its date and precision.

The primary mCRPC date is the earliest of
  both_hold  the first date both metastatic disease and CRPC held
  statement  a dated gated mcrpc_statement

A start of a drug on the proxy allowlist (cabazitaxel, radium-223,
sipuleucel-T, from MEDICATIONS or a stated note start) is a separate
`drug_proxy` basis: it supplies the date when no primary basis exists, and caps
the date (proxy_precedes_primary=True) when it is certainly earlier. The primary
date and basis stay in their own columns. Lu-177-PSMA is NOT on the allowlist:
it is now also indicated in metastatic hormone-sensitive disease. Its first
start is recorded for audit only.

Bump MCRPC_CRITERIA_VERSION or PROXY_POLICY_VERSION on any change.
"""

from datetime import timedelta

from tasks.disease_course.psa import merge_psa, pcwg3_progression
from tasks.disease_course.therapy import (
    classify_agent,
    episode_starts,
    exposure_at,
    patient_intervals,
    stop_points,
)
from tasks.onset import both_hold, date_interval, earliest

MCRPC_CRITERIA_VERSION = "mcrpc-criteria-v1"
PROXY_POLICY_VERSION = "mcrpc-proxy-v1"
PROXY_AGENTS = frozenset({"cabazitaxel", "radium", "sipuleucel"})
CASTRATE_NG_DL = 50.0
TESTOSTERONE_WINDOW_DAYS = 60
SUPPORTING_TIERS = {"supported", "reported"}

RULE_DEFAULTS = {
    "mcrpc_criteria_version": MCRPC_CRITERIA_VERSION,
    "proxy_policy_version": PROXY_POLICY_VERSION,
    "proxy_agents": sorted(PROXY_AGENTS),
    "proxy_excluded": ["lutetium_psma"],
    "castrate_ng_dl": CASTRATE_NG_DL,
    "testosterone_window_days": TESTOSTERONE_WINDOW_DAYS,
    "assumed_castration_requires": sorted(SUPPORTING_TIERS),
}


def _source(row):
    return f"note:{row['note_uid']}:{row['finding_index']}"


def _point(when, precision, basis, source, quote=None, **extra):
    return {"event_date": when, "date_precision": precision, "source_note_date": None,
            "_basis": basis, "_source": source, "quote": quote, **extra}


def _castrate(row):
    """A testosterone result certainly below the castrate level."""
    if row["comparator"] == "undetectable":
        return True
    value = row["testosterone_ng_dl"]
    if value is None:
        return False
    if row["comparator"] == "<":
        return value <= CASTRATE_NG_DL
    return row["comparator"] in {"=", "<="} and value < CASTRATE_NG_DL


def castration_tier(testosterone_rows, rise, confirmation, intervals, stops):
    """("measured" | "assumed" | None, reason, testosterone dates used)."""
    lo = rise["date"] - timedelta(days=TESTOSTERONE_WINDOW_DAYS)
    hi = rise["date"] + timedelta(days=TESTOSTERONE_WINDOW_DAYS)
    window = [
        r for r in testosterone_rows
        if lo.isoformat() <= r["date"] <= hi.isoformat()
    ]
    dates = sorted({r["date"] for r in window})
    high = [
        r for r in window
        if r["testosterone_ng_dl"] is not None and r["comparator"] in {"=", ">", ">="}
        and r["testosterone_ng_dl"] >= CASTRATE_NG_DL
    ]
    if high:
        return None, "testosterone_not_castrate", dates
    if any(_castrate(r) for r in window):
        return "measured", None, dates
    at_rise = exposure_at(intervals, rise["date"], stops)
    at_confirm = exposure_at(intervals, confirmation["date"], stops)
    if at_rise["adt_certainty"] in SUPPORTING_TIERS and at_confirm["adt_certainty"] in SUPPORTING_TIERS:
        return "assumed", None, dates
    if at_rise["adt_certainty"] == "estimated" or at_confirm["adt_certainty"] == "estimated":
        return None, "adt_estimated_only", dates
    return None, "adt_not_documented", dates


def derived_crpc(rows, psa_rows, med_rows, testosterone_rows):
    """(earliest castrate PCWG3 point or None, reasons for episodes that did not count)."""
    measurements, _ = merge_psa(psa_rows, rows)
    episodes = episode_starts(med_rows, rows)
    intervals = patient_intervals(med_rows, rows)
    stops = stop_points(rows)
    points, reasons = [], []
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
        tier, reason, dates = castration_tier(
            testosterone_rows, result["event"], result["confirmation"], intervals, stops
        )
        if tier is None:
            reasons.append(f"{label}:{reason}")
            continue
        exposure = exposure_at(intervals, result["event"]["date"], stops)
        points.append(_point(
            result["date"], result["precision"], "pcwg3", result["event"]["source_id"],
            castration_tier=tier, testosterone_dates=dates,
            exposure_basis=exposure["adt_basis"], exposure_certainty=exposure["adt_certainty"],
            confirmation_date=result["confirmation"]["date"].isoformat(),
            episode=label,
        ))
    return earliest(points), reasons


def proxy_starts(rows, med_rows):
    """(earliest allowlisted drug start, first Lu-PSMA start) as points."""
    proxies, lu = [], []
    for record in med_rows:
        when = record.get("administration_date") or record.get("date")
        point = _point(when, "day", "drug_proxy", record["source_id"], agent=record["agent"],
                       evidence="medications")
        if record["agent"] in PROXY_AGENTS:
            proxies.append(point)
        elif record["drug_class"] == "lu_psma":
            lu.append(point)
    for row in rows:
        if row["item"] != "therapy_start" or row.get("date_source") != "stated":
            continue
        match = classify_agent(row.get("agent")) or classify_agent(row.get("quote"))
        if match is None:
            continue
        point = _point(row["event_date"], row["date_precision"], "drug_proxy", _source(row),
                       quote=row["quote"], agent=match[1], evidence="note")
        if match[1] in PROXY_AGENTS:
            proxies.append(point)
        elif match[0] == "lu_psma":
            lu.append(point)
    return earliest(proxies), earliest(lu)


def _certainly_before(a, b):
    return date_interval(a["event_date"], a["date_precision"])[1] < date_interval(
        b["event_date"], b["date_precision"])[0]


def met_point(met):
    """Upstream metastatic label {"date", "precision", "source"} as a point, or None."""
    if not met or not met.get("date"):
        return None
    return _point(met["date"], met.get("precision") or "unknown", "metastatic", met.get("source"))


def evaluate_mcrpc(rows, psa_rows, med_rows, testosterone_rows, met):
    """Label fields for one patient. `met` is the upstream metastatic label or None."""
    statements = [
        _point(r["event_date"], r["date_precision"], "statement", _source(r), quote=r["quote"])
        for r in rows if r["item"] in {"crpc_statement", "mcrpc_statement"}
    ]
    mcrpc_statement = earliest([
        _point(r["event_date"], r["date_precision"], "statement", _source(r), quote=r["quote"])
        for r in rows if r["item"] == "mcrpc_statement"
    ])
    derived, reasons = derived_crpc(rows, psa_rows, med_rows, testosterone_rows)
    crpc = earliest(statements + ([derived] if derived else []))
    metastatic = met_point(met)

    primaries = []
    if crpc and metastatic:
        joint = both_hold([metastatic, crpc])
        if joint:
            primaries.append(_point(joint["event_date"], joint["date_precision"], "both_hold", None))
    if mcrpc_statement:
        primaries.append(mcrpc_statement)
    primary = earliest(primaries)
    proxy, lu = proxy_starts(rows, med_rows)

    precedes = False
    if primary is None:
        final = proxy
    elif proxy is not None and _certainly_before(proxy, primary):
        final, precedes = proxy, True
    else:
        final = primary
    return {
        "is_mcrpc": final is not None,
        "mcrpc_date": final and final["event_date"],
        "mcrpc_date_precision": final and final["date_precision"],
        "mcrpc_basis": final and final["_basis"],
        "proxy_precedes_primary": precedes,
        "primary_date": primary and primary["event_date"],
        "primary_date_precision": primary and primary["date_precision"],
        "primary_basis": primary and primary["_basis"],
        "met_date": metastatic and metastatic["event_date"],
        "met_date_precision": metastatic and metastatic["date_precision"],
        "met_label_source": met and met.get("source"),
        "crpc_date": crpc and crpc["event_date"],
        "crpc_date_precision": crpc and crpc["date_precision"],
        "crpc_basis": crpc and crpc["_basis"],
        "crpc_source": crpc and crpc["_source"],
        "crpc_quote": crpc and crpc.get("quote"),
        "crpc_castration_tier": crpc and crpc.get("castration_tier"),
        "crpc_testosterone_dates": (crpc or {}).get("testosterone_dates") or [],
        "crpc_exposure_basis": crpc and crpc.get("exposure_basis"),
        "crpc_exposure_certainty": crpc and crpc.get("exposure_certainty"),
        "crpc_confirmation_date": crpc and crpc.get("confirmation_date"),
        "pcwg3_unevaluable": reasons,
        "mcrpc_statement_date": mcrpc_statement and mcrpc_statement["event_date"],
        "mcrpc_statement_precision": mcrpc_statement and mcrpc_statement["date_precision"],
        "mcrpc_statement_quote": mcrpc_statement and mcrpc_statement["quote"],
        "proxy_date": proxy and proxy["event_date"],
        "proxy_date_precision": proxy and proxy["date_precision"],
        "proxy_agent": proxy and proxy["agent"],
        "proxy_evidence": proxy and proxy["evidence"],
        "proxy_source": proxy and proxy["_source"],
        "lu_psma_date": lu and lu["event_date"],
    }
