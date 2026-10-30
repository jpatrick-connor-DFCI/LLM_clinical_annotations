"""Patient-level course facts from gated course findings: local therapy and index dates.

Shared by the course labels (run_course_extraction.build_labels) and the BCR,
progression and mCRPC builders, so all of them agree on RP and RT dates.

Index dates come only from dates written in the text (date_source "stated"):
a note-date fallback would date "s/p RALP" to whenever the note was written.
A procedure documented without a date still sets `local_therapy`.

Primary RT is radiation with intent "primary", or with intent "unknown" when no
prostatectomy is documented and the target (or quote) names the prostate and no
bed, nodal-only or metastatic site. Salvage RT covers "salvage" and "adjuvant".
"""

import re

from tasks.disease_course.therapy import CASTRATING_CLASSES, classify_agent
from tasks.onset import earliest

PROSTATE_TARGET = re.compile(r"\bprostat\w*|\bseminal\b|\bbrachy\w*|\bseeds?\b", re.IGNORECASE)
NON_PRIMARY_TARGET = re.compile(
    r"\b(?:bed|fossa|vertebra\w*|spine|spinal|rib|ribs|femur|femoral|humer\w*|sacr\w*|"
    r"ilium|acetabul\w*|skull|brain|lung|liver|bone|bony|osseous|metasta\w*|oligo\w*|"
    r"lesion|node|nodal|lymph|[TLC]\d{1,2})\b",
    re.IGNORECASE,
)


def _stated(rows):
    return [r for r in rows if r.get("date_source") == "stated" and r.get("event_date")]


def _date_fields(prefix, row):
    return {
        f"{prefix}_date": row and row["event_date"],
        f"{prefix}_date_precision": row and row["date_precision"],
        f"{prefix}_date_source": row and f"note:{row['note_uid']}:{row['finding_index']}",
        f"{prefix}_quote": row and row["quote"],
    }


def _is_primary_rt(row, rp_documented):
    intent = row.get("rt_intent")
    if intent == "primary":
        return True
    if intent not in (None, "unknown") or rp_documented:
        return False
    target = " ".join(filter(None, (row.get("rt_target"), row.get("quote"))))
    return bool(PROSTATE_TARGET.search(target)) and not NON_PRIMARY_TARGET.search(
        row.get("rt_target") or ""
    )


def local_therapy(rows):
    """RP / primary RT / salvage RT index facts from one patient's usable findings."""
    rp_rows = [r for r in rows if r["item"] == "radical_prostatectomy"]
    rt_rows = [r for r in rows if r["item"] == "radiation_therapy"]
    rp_documented = bool(rp_rows)
    primary = [r for r in rt_rows if _is_primary_rt(r, rp_documented)]
    salvage = [r for r in rt_rows if r.get("rt_intent") in {"salvage", "adjuvant"}]
    rp = earliest(_stated(rp_rows))
    primary_rt = earliest(_stated(primary))
    salvage_rt = earliest(_stated(salvage))
    adt_starts = [
        r for r in _stated([r for r in rows if r["item"] == "therapy_start"])
        if (classify_agent(r.get("agent")) or (None,))[0] in CASTRATING_CLASSES
    ]
    first_adt = earliest(adt_starts)
    if rp_documented and primary:
        kind = "rp_and_rt"
    elif rp_documented:
        kind = "rp"
    elif primary:
        kind = "rt"
    else:
        kind = None
    return {
        "local_therapy": kind,
        **_date_fields("rp", rp),
        **_date_fields("primary_rt", primary_rt),
        **_date_fields("salvage_rt", salvage_rt),
        "salvage_rt_intent": salvage_rt and salvage_rt.get("rt_intent"),
        **_date_fields("first_note_adt", first_adt),
        "_rp": rp,
        "_primary_rt": primary_rt,
    }


def index_point(row):
    """{"event_date", "date_precision"} for a finding row, or None."""
    if row is None:
        return None
    return {"event_date": row["event_date"], "date_precision": row["date_precision"],
            "source_note_date": row.get("source_note_date")}
