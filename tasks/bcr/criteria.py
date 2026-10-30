"""Biochemical recurrence for one patient from gated course findings and LABS PSA.

Bases, the earliest eligible one winning:

  psa_aua      after radical prostatectomy: documented clearance (< 0.1 or
               undetectable), then PSA >= 0.2 confirmed by a later separate
               draw >= 0.2. Without clearance the basis is unevaluable, so
               persistent PSA alone is never BCR.
  psa_phoenix  after primary radiation (no prostatectomy): PSA >= nadir + 2.
  statement    a gated "biochemical recurrence" statement (bcr_kind
               "recurrence"), unless it is certainly dated before the local
               therapy. A "persistent PSA" statement only sets the
               persistence flag.
  imaging      a gated definite or probable imaging recurrence certainly after
               the local therapy. With persistent PSA and no clearance before
               the image, it shows residual disease, not a new recurrence.

PSA and imaging bases need a dated local therapy; a statement does not.
`bcr` is True with any basis, False only when the PSA rule was evaluable and not
met and nothing else applies, otherwise unknown.

Bump BCR_CRITERIA_VERSION on any change; label metadata records it.
"""

from tasks.disease_course.course import index_point, local_therapy
from tasks.disease_course.psa import (
    aua_bcr,
    merge_psa,
    persistent_psa,
    phoenix_bcr,
    psa_clearance,
    summary,
)
from tasks.onset import date_interval, earliest

BCR_CRITERIA_VERSION = "bcr-criteria-v1"
IMAGING_CERTAINTIES = {"definite", "probable"}


def _source(row):
    return f"note:{row['note_uid']}:{row['finding_index']}"


def _bounds(point):
    return date_interval(point["event_date"], point.get("date_precision") or "day")


def _measurement_point(m, basis):
    return {
        "event_date": m["date"].isoformat(), "date_precision": m["precision"],
        "source_note_date": None, "_basis": basis, "_source": m["source_id"],
    }


def evaluate_bcr(rows, psa_rows):
    """Label fields for one patient. `rows` are gated course findings."""
    facts = local_therapy(rows)
    rp, primary_rt = facts["_rp"], facts["_primary_rt"]
    kind = facts["local_therapy"]
    measurements, audit = merge_psa(psa_rows, rows)

    if kind in {"rp", "rp_and_rt"}:
        index, rule = rp, "aua"
    elif kind == "rt":
        index, rule = primary_rt, "phoenix"
    else:
        index, rule = None, None
    index_pt = index_point(index)

    persistence = persistent_psa(measurements, index_point(rp)) if rp else None
    clearance = psa_clearance(measurements, index_point(rp)) if rp else None
    persistent_statements = [
        r for r in rows if r["item"] == "bcr_statement" and r.get("bcr_kind") == "persistent_psa"
    ]
    persistent_flag = persistence["status"] if persistence else None
    persistent_basis = "psa" if persistent_flag else None
    if persistent_statements and not persistent_flag:
        persistent_flag, persistent_basis = True, "statement"

    if rule == "aua":
        psa = aua_bcr(measurements, index_pt)
    elif rule == "phoenix":
        psa = phoenix_bcr(measurements, index_pt)
    else:
        psa = {"status": "unevaluable", "reason": "no_local_therapy"}

    candidates = []
    if psa["status"] == "met":
        candidates.append(_measurement_point(psa["event"], f"psa_{rule}"))

    statements = []
    for row in rows:
        if row["item"] != "bcr_statement" or row.get("bcr_kind") == "persistent_psa":
            continue
        if index_pt and _bounds(row)[1] < _bounds(index_pt)[0]:
            continue
        statements.append({**row, "_basis": "statement", "_source": _source(row)})
    statement = earliest(statements)
    if statement:
        candidates.append(statement)

    imaging, excluded = None, []
    if index_pt:
        images = []
        clear_hi = clearance["hi"] if clearance else None
        for row in rows:
            if row["item"] != "imaging_recurrence" or row.get("certainty") not in IMAGING_CERTAINTIES:
                continue
            lo, hi = _bounds(row)
            if lo <= _bounds(index_pt)[1]:
                excluded.append(f"{_source(row)}:not_after_index")
                continue
            if persistent_flag and (clear_hi is None or clear_hi >= lo):
                excluded.append(f"{_source(row)}:residual_after_persistent_psa")
                continue
            images.append({**row, "_basis": "imaging", "_source": _source(row)})
        imaging = earliest(images)
        if imaging:
            candidates.append(imaging)

    first = earliest(candidates)
    if first is not None:
        bcr = True
    elif psa["status"] == "not_met":
        bcr = False
    else:
        bcr = None
    event = psa.get("event")
    confirmation = psa.get("confirmation")
    return {
        "bcr": bcr,
        "bcr_date": first and first["event_date"],
        "bcr_date_precision": first and first["date_precision"],
        "bcr_basis": first and first["_basis"],
        "bcr_bases": sorted({c["_basis"] for c in candidates}),
        "bcr_source": first and first["_source"],
        "local_therapy": kind,
        "index_date": index_pt and index_pt["event_date"],
        "index_date_precision": index_pt and index_pt["date_precision"],
        "persistent_psa": persistent_flag,
        "persistent_psa_basis": persistent_basis,
        "persistent_psa_reason": persistence and persistence["reason"],
        "psa_clearance_date": clearance and clearance["date"].isoformat(),
        "psa_bcr_rule": rule,
        "psa_bcr_status": psa["status"],
        "psa_bcr_reason": psa.get("reason"),
        "psa_bcr_date": event and event["date"].isoformat(),
        "psa_bcr_value": event and event["value"],
        "psa_bcr_confirmation_date": confirmation and confirmation["date"].isoformat(),
        "psa_bcr_source_ids": sorted(
            set((event or {}).get("sources", [])) | set((confirmation or {}).get("sources", []))
        ),
        "phoenix_nadir": (psa.get("nadir") and summary(psa["nadir"])["value"]),
        "statement_date": statement and statement["event_date"],
        "statement_date_precision": statement and statement["date_precision"],
        "statement_quote": statement and statement["quote"],
        "imaging_date": imaging and imaging["event_date"],
        "imaging_date_precision": imaging and imaging["date_precision"],
        "imaging_certainty": imaging and imaging.get("certainty"),
        "imaging_modality": imaging and imaging.get("modality"),
        "imaging_site": imaging and imaging.get("site_category"),
        "imaging_quote": imaging and imaging["quote"],
        "imaging_exclusions": excluded,
        "n_psa_measurements": len(measurements),
        "n_psa_conflicts": sum(1 for m in measurements if m["conflict"]),
        "psa_audit": [f"{a['source_id']}:{a['reason']}" for a in audit],
    }
