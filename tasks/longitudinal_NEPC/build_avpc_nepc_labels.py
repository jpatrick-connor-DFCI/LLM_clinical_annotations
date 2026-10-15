"""Reduce the AVPC/NEPC criteria timeline into one patient-level label row.

This is a pure "read one parquet, reduce, write one parquet" script: no LLM
calls, no resume fingerprint, no chunk log. It reads
`avpc_nepc_timeline.parquet` (built by `build_nepc_timeline.py`) and derives,
per patient:

- `AVPC`: Aparicio et al. 2013, Table 1 -- castration-resistant, metastatic
  prostate carcinoma with at least one of the `C1`-`C7` criteria. Metastatic
  disease is a `STATE:metastatic` row or C2/C3/C5, which presuppose it;
  castration resistance is a `STATE:castration_resistant` row or C7. Small-cell
  histology (C1) is exempt from castration resistance (Table 1, footnote a)
  but must still be metastatic. Timed at the earliest `event_date` by which a
  criterion and both disease states are all documented.
- `NEPC_TIMELINE`: any `NEPC:*` criterion, timed at the earliest such
  `event_date`.
- `AVPC_NEPC` (modeled): the union of the two, with NEPC-precedence timing --
  `nepc_timeline_date` if any NEPC criterion is present, else `avpc_date`.

`metastatic_date` / `castration_resistant_date` are each disease state's
earliest dated documentation, and `avpc_gate_missing` lists the states a
patient with a dated criterion lacks.

The timeline is already deduped to earliest onset per (patient, criterion) by
`build_timeline`'s `_prefer_onset`, so no re-deduplication happens here.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.parquet_io import write_rows_atomic  # noqa: E402
from tasks.longitudinal_NEPC.build_nepc_timeline import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    STATE_KEYS,
    TIMELINE_COLUMNS,
    VALID_CRITERIA,
)
from tasks.longitudinal_NEPC.mcrpc_gate import (  # noqa: E402
    CASTRATION_EXEMPT_CRITERIA,
    CASTRATION_RESISTANT,
    CASTRATION_RESISTANT_CRITERIA,
    METASTATIC,
    METASTATIC_CRITERIA,
)

AVPC_KEYS = {f"C{i}" for i in range(1, 8)}
NEPC_KEYS = {key for key in VALID_CRITERIA if key.startswith("NEPC:")}
# Table 1: "castrate-resistant prostate carcinoma with at least 1 of" C1-C7.
AVPC_MIN_CRITERIA = 1
METASTATIC_KEYS = {f"STATE:{METASTATIC}"} | METASTATIC_CRITERIA
CASTRATION_RESISTANT_KEYS = (
    {f"STATE:{CASTRATION_RESISTANT}"} | CASTRATION_RESISTANT_CRITERIA
)
# Rows that satisfy the castration-resistance condition: resistance itself, or
# small-cell histology, which the definition exempts from it.
CASTRATION_GATE_KEYS = CASTRATION_RESISTANT_KEYS | CASTRATION_EXEMPT_CRITERIA

LABEL_COLUMNS = [
    "DFCI_MRN",
    "has_avpc",
    "avpc_date",
    "has_nepc_timeline",
    "nepc_timeline_date",
    "has_avpc_nepc",
    "avpc_nepc_date",
    "date_source",
    "date_precision",
    "n_avpc_criteria",
    "avpc_criteria",
    "nepc_criteria",
    "supporting_quote",
    "confidence",
    "source_note_date",
    "label_source",
    "metastatic_date",
    "castration_resistant_date",
    "avpc_gate_missing",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reduce the AVPC/NEPC criteria timeline into one patient-level "
            "label row (castration-resistant metastatic disease with >=1 "
            "Aparicio criterion and/or any NEPC feature, with NEPC-precedence "
            "timing)."
        )
    )
    parser.add_argument(
        "--timeline-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "avpc_nepc_timeline.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "avpc_nepc_labels.parquet",
    )
    return parser.parse_args()


def _read_timeline(timeline_path):
    if not timeline_path.exists() or timeline_path.stat().st_size == 0:
        return pl.DataFrame(
            schema={column: pl.String for column in TIMELINE_COLUMNS}
        )
    return pl.read_parquet(timeline_path)


def _earliest(rows, keys):
    """Return the earliest dated row whose criterion is in `keys`, else None."""
    return min(
        (row for row in rows if row["criterion_added"] in keys),
        key=lambda row: (row["event_date"], row["criterion_added"]),
        default=None,
    )


def _avpc_definition_met(criteria):
    """Whether a set of timeline criterion keys meets the AVPC definition."""
    return (
        len(criteria & AVPC_KEYS) >= AVPC_MIN_CRITERIA
        and bool(criteria & METASTATIC_KEYS)
        and bool(criteria & CASTRATION_GATE_KEYS)
    )


def _build_patient_label(mrn, rows):
    """Reduce one patient's timeline rows into a label dict.

    `rows` is every timeline row for this patient (dated criteria and disease
    states, undated ones, and/or the synthetic `conventional` row). Returns a
    dict shaped like `LABEL_COLUMNS`, or `None` if the patient has no rows at
    all (should not happen given how callers group, but keeps this function
    total).
    """
    if not rows:
        return None

    dated = [row for row in rows if row.get("event_date")]
    undated = [row for row in rows if not row.get("event_date")]
    # Disease-state rows sit alongside the conventional row of a patient with
    # no criteria (most mCRPC is conventional), so they don't count here.
    criterion_rows = [
        row for row in rows if row.get("criterion_added") not in STATE_KEYS
    ]
    is_conventional_only = (
        len(criterion_rows) == 1
        and criterion_rows[0].get("criterion_added") == "conventional"
    )

    dated_sorted = sorted(
        dated, key=lambda row: (row["event_date"], row["criterion_added"])
    )

    # Walk dated rows in (event_date, criterion_added) order, accumulating
    # distinct C1-C7 criteria. criteria_row is the row at which the cumulative
    # C-only count first reaches AVPC_MIN_CRITERIA. Same-date blocks are
    # applied together (matching build_timeline's own same-date
    # cumulative-count semantics) so a block that pushes the count from below
    # AVPC_MIN_CRITERIA to at or above it in one date is dated at that date.
    c_seen = set()
    criteria_row = None
    position = 0
    while position < len(dated_sorted):
        event_date = dated_sorted[position]["event_date"]
        same_date = []
        while (
            position < len(dated_sorted)
            and dated_sorted[position]["event_date"] == event_date
        ):
            same_date.append(dated_sorted[position])
            position += 1
        before = len(c_seen)
        for row in same_date:
            if row["criterion_added"] in AVPC_KEYS:
                c_seen.add(row["criterion_added"])
        if criteria_row is None and before < AVPC_MIN_CRITERIA <= len(c_seen):
            # The earliest-sorted C-key row in the block; one exists because the
            # count only grows via AVPC_KEYS rows.
            criteria_row = next(
                r for r in same_date if r["criterion_added"] in AVPC_KEYS
            )
    n_avpc_criteria = len(c_seen)

    # The definition presupposes castration-resistant metastatic disease, so a
    # criterion alone is not AVPC. Each disease state is timed at its earliest
    # dated documentation, whether a STATE:* row or a criterion implying it.
    metastatic_row = _earliest(dated, METASTATIC_KEYS)
    resistant_row = _earliest(dated, CASTRATION_RESISTANT_KEYS)
    castration_gate_row = _earliest(dated, CASTRATION_GATE_KEYS)
    avpc_gate_missing = []
    if criteria_row is not None:
        if metastatic_row is None:
            avpc_gate_missing.append(METASTATIC)
        if castration_gate_row is None:
            avpc_gate_missing.append(CASTRATION_RESISTANT)
    avpc_row = None
    if criteria_row is not None and not avpc_gate_missing:
        # AVPC begins once all three hold. max() keeps the first of tied rows,
        # so on a tie the criterion row, not a disease-state row, is the
        # defining row and supplies the quote.
        avpc_row = max(
            (criteria_row, metastatic_row, castration_gate_row),
            key=lambda row: row["event_date"],
        )
    avpc_date = avpc_row["event_date"] if avpc_row is not None else None

    # Undated evidence can never time AVPC, but it can complete the
    # definition -- used only to detect the undated-only-positive demotion case
    # below, not for n_avpc_criteria (spec: "max C-only cumulative count
    # reached ... over dated rows only").
    undated_only_avpc_positive = avpc_date is None and _avpc_definition_met(
        {row.get("criterion_added") for row in rows}
    )

    # NEPC: earliest event_date among dated NEPC:* rows. NEPC:* rows never
    # contribute to the AVPC C-count above (AVPC_KEYS excludes them).
    nepc_dated = sorted(
        (row for row in dated if row["criterion_added"] in NEPC_KEYS),
        key=lambda row: row["event_date"],
    )
    nepc_undated_only = (
        not nepc_dated
        and any(row["criterion_added"] in NEPC_KEYS for row in undated)
    )
    nepc_timeline_date = nepc_dated[0]["event_date"] if nepc_dated else None
    nepc_row = nepc_dated[0] if nepc_dated else None
    has_nepc_timeline = 1 if nepc_dated else 0

    has_avpc = 1 if avpc_date is not None else 0

    # NEPC precedence for timing: nepc_timeline_date wins whenever any NEPC
    # criterion is present, regardless of whether AVPC was also met earlier.
    if has_nepc_timeline:
        avpc_nepc_date = nepc_timeline_date
        defining_row = nepc_row
    elif has_avpc:
        avpc_nepc_date = avpc_date
        defining_row = avpc_row
    else:
        avpc_nepc_date = None
        defining_row = None

    has_avpc_nepc = 1 if (has_avpc or has_nepc_timeline) else 0

    # Undated-only-positive demotion: a patient who meets the AVPC definition
    # only once undated evidence is counted, and who has no dated/undated NEPC
    # evidence either, is demoted to negative.
    # A patient with dated NEPC evidence is unaffected (NEPC already timed).
    # A patient with *only* undated NEPC evidence and no dated AVPC/NEPC
    # evidence is also demoted.
    # has_avpc / has_nepc_timeline are both dated by construction (avpc_date
    # and nepc_timeline_date only ever come from dated rows), so the demotion
    # check only fires when neither is already true.
    demoted_undated_only = False
    if not has_avpc and not has_nepc_timeline and (
        undated_only_avpc_positive or nepc_undated_only
    ):
        demoted_undated_only = True
        has_avpc_nepc = 0

    avpc_criteria = sorted(c_seen)
    nepc_criteria = sorted(
        {row["criterion_added"] for row in dated if row["criterion_added"] in NEPC_KEYS}
    )

    if is_conventional_only:
        label_source = "conventional"
    elif has_avpc_nepc:
        label_source = "timeline_positive"
    else:
        label_source = "timeline_negative"

    if defining_row is not None:
        date_source = defining_row.get("date_source")
        date_precision = defining_row.get("date_precision")
        supporting_quote = defining_row.get("supporting_quote")
        confidence = defining_row.get("confidence")
        source_note_date = defining_row.get("source_note_date")
    else:
        date_source = None
        date_precision = "unknown"
        supporting_quote = None
        confidence = None
        source_note_date = None

    return {
        "DFCI_MRN": mrn,
        "has_avpc": has_avpc,
        "avpc_date": avpc_date,
        "has_nepc_timeline": has_nepc_timeline,
        "nepc_timeline_date": nepc_timeline_date,
        "has_avpc_nepc": has_avpc_nepc,
        "avpc_nepc_date": avpc_nepc_date,
        "date_source": date_source,
        "date_precision": date_precision,
        "n_avpc_criteria": n_avpc_criteria,
        "avpc_criteria": avpc_criteria,
        "nepc_criteria": nepc_criteria,
        "supporting_quote": supporting_quote,
        "confidence": confidence,
        "source_note_date": source_note_date,
        "label_source": label_source,
        "metastatic_date": (
            metastatic_row["event_date"] if metastatic_row is not None else None
        ),
        "castration_resistant_date": (
            resistant_row["event_date"] if resistant_row is not None else None
        ),
        "avpc_gate_missing": avpc_gate_missing,
        "_demoted_undated_only": demoted_undated_only,
    }


def build_labels(timeline_path, labels_path):
    """Read the timeline, reduce to one label row per patient, write it out.

    Returns the number of label rows written.
    """
    timeline = _read_timeline(timeline_path)

    by_patient = {}
    for row in timeline.iter_rows(named=True):
        mrn = row.get("DFCI_MRN")
        if mrn is None:
            continue
        by_patient.setdefault(mrn, []).append(row)

    rows = []
    n_demoted = 0
    for mrn in sorted(by_patient):
        label = _build_patient_label(mrn, by_patient[mrn])
        if label is None:
            continue
        if label.pop("_demoted_undated_only"):
            n_demoted += 1
        rows.append(label)

    rows.sort(key=lambda row: row["DFCI_MRN"])
    write_rows_atomic(labels_path, rows, LABEL_COLUMNS)

    _print_summary(rows, n_demoted)
    return len(rows)


def _print_summary(rows, n_demoted):
    n_patients = len(rows)
    n_avpc = sum(row["has_avpc"] for row in rows)
    n_nepc = sum(row["has_nepc_timeline"] for row in rows)
    n_union = sum(row["has_avpc_nepc"] for row in rows)
    n_exempt = sum(
        1
        for row in rows
        if row["has_avpc"] and row["castration_resistant_date"] is None
    )

    print(f"Wrote {n_patients} patient labels")
    print(f"  AVPC-positive (has_avpc=1): {n_avpc}")
    if n_exempt:
        print(
            f"    via small-cell histology without documented castration "
            f"resistance: {n_exempt}"
        )
    print(f"  NEPC-positive (has_nepc_timeline=1): {n_nepc}")
    print(f"  AVPC_NEPC union positive (has_avpc_nepc=1): {n_union}")
    if n_demoted:
        print(
            f"  Demoted to negative (definition only met via undated "
            f"evidence): {n_demoted}"
        )

    gate_failures = Counter(
        " + ".join(row["avpc_gate_missing"])
        for row in rows
        if row["avpc_gate_missing"]
    )
    if gate_failures:
        print(
            "  Dated criterion but not castration-resistant metastatic "
            f"disease (has_avpc=0): {sum(gate_failures.values())}"
        )
        for missing, count in gate_failures.most_common():
            print(f"    missing {missing}: {count}")

    criteria_counts = Counter(row["n_avpc_criteria"] for row in rows)
    print("  n_avpc_criteria distribution:")
    for count in sorted(criteria_counts):
        print(f"    {count}: {criteria_counts[count]}")

    event_rows = [row for row in rows if row["has_avpc_nepc"]]
    if event_rows:
        source_counts = Counter(row["date_source"] for row in event_rows)
        precision_counts = Counter(row["date_precision"] for row in event_rows)
        print("  date_source breakdown among events:")
        for source, count in source_counts.most_common():
            print(f"    {source}: {count}")
        print("  date_precision breakdown among events:")
        for precision, count in precision_counts.most_common():
            print(f"    {precision}: {count}")
        if source_counts.get("note_date"):
            print(
                "  NOTE: date_source == 'note_date' means earliest "
                "*documentation*, not onset -- this is more common here than "
                "in the strict nepc_dx_labels endpoint, since per-criterion "
                "mentions rarely carry a stated onset date."
            )


def run(args):
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    count = build_labels(args.timeline_path, args.output_path)
    print(f"Wrote AVPC/NEPC labels ({count} rows): {args.output_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
