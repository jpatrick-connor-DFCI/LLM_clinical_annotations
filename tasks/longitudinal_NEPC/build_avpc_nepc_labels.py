"""Reduce the AVPC/NEPC criteria timeline into one patient-level label row.

This is a pure "read one parquet, reduce, write one parquet" script: no LLM
calls, no resume fingerprint, no chunk log. It reads
`avpc_nepc_timeline.parquet` (built by `build_nepc_timeline.py`) and derives,
per patient:

- `AVPC`: >=3 distinct Aparicio `C1`-`C7` criteria, timed at the earliest
  `event_date` the count first reaches 3.
- `NEPC_TIMELINE`: any `NEPC:*` criterion, timed at the earliest such
  `event_date`.
- `AVPC_NEPC` (modeled): the union of the two, with NEPC-precedence timing --
  `nepc_timeline_date` if any NEPC criterion is present, else `avpc_date`.

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
    TIMELINE_COLUMNS,
    VALID_CRITERIA,
)

AVPC_KEYS = {f"C{i}" for i in range(1, 8)}
NEPC_KEYS = {key for key in VALID_CRITERIA if key.startswith("NEPC:")}
AVPC_THRESHOLD = 3

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
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reduce the AVPC/NEPC criteria timeline into one patient-level "
            "label row (>=3 Aparicio criteria and/or any NEPC feature, with "
            "NEPC-precedence timing)."
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


def _build_patient_label(mrn, rows):
    """Reduce one patient's timeline rows into a label dict.

    `rows` is every timeline row for this patient (dated criteria, undated
    criteria, and/or the synthetic `conventional` row). Returns a dict shaped
    like `LABEL_COLUMNS`, or `None` if the patient has no rows at all (should
    not happen given how callers group, but keeps this function total).
    """
    if not rows:
        return None

    dated = [row for row in rows if row.get("event_date")]
    undated = [row for row in rows if not row.get("event_date")]
    is_conventional_only = len(rows) == 1 and rows[0].get("criterion_added") == "conventional"

    dated_sorted = sorted(
        dated, key=lambda row: (row["event_date"], row["criterion_added"])
    )

    # Walk dated rows in (event_date, criterion_added) order, accumulating
    # distinct C1-C7 criteria. avpc_date is the event_date of the row at
    # which the cumulative C-only count first reaches AVPC_THRESHOLD.
    # Same-date blocks are applied together (matching build_timeline's own
    # same-date cumulative-count semantics) so a block that pushes the count
    # from 1 to 3 in one date is dated at that date.
    c_seen = set()
    avpc_date = None
    avpc_row = None
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
        if avpc_date is None and before < AVPC_THRESHOLD <= len(c_seen):
            avpc_date = event_date
            # The row that "defined" this block: earliest-sorted C-key row in
            # the block that is itself an AVPC criterion (falls back to the
            # first row in the block if none, which cannot happen since the
            # count only grows via AVPC_KEYS rows).
            avpc_candidates = [r for r in same_date if r["criterion_added"] in AVPC_KEYS]
            avpc_row = avpc_candidates[0] if avpc_candidates else same_date[0]
    n_avpc_criteria = len(c_seen)

    # Undated evidence can never time or count toward the threshold, but it
    # can still push the *set* of distinct criteria the patient has ever
    # shown -- used only to detect the undated-only-positive demotion case
    # below, not for n_avpc_criteria (spec: "max C-only cumulative count
    # reached ... over dated rows only").
    c_seen_including_undated = set(c_seen)
    for row in undated:
        if row["criterion_added"] in AVPC_KEYS:
            c_seen_including_undated.add(row["criterion_added"])
    undated_only_avpc_positive = (
        avpc_date is None and len(c_seen_including_undated) >= AVPC_THRESHOLD
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
    # criterion is present, regardless of whether AVPC also crossed threshold
    # earlier.
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

    # Undated-only-positive demotion: a patient whose AVPC threshold is
    # reached only via undated evidence (no dated row ever reaches 3), and
    # who has no dated/undated NEPC evidence either, is demoted to negative.
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

    print(f"Wrote {n_patients} patient labels")
    print(f"  AVPC-positive (has_avpc=1): {n_avpc}")
    print(f"  NEPC-positive (has_nepc_timeline=1): {n_nepc}")
    print(f"  AVPC_NEPC union positive (has_avpc_nepc=1): {n_union}")
    if n_demoted:
        print(
            f"  Demoted to negative (threshold only reached via undated "
            f"evidence): {n_demoted}"
        )

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
