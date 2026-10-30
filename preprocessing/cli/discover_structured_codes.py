"""Print the structured codes and exposure fields the disease-course builders rely on.

Run on the cluster before the first builder run to confirm:
  - which LABS TEST_TYPE_CD / TEST_TYPE_DESCR / RESULT_UOM_NM combinations look
    like PSA or testosterone (compare with preprocessing.structured.PSA_CODES and
    TESTOSTERONE_CODES);
  - which medication names map to each drug class, and which do not map;
  - for each ADT agent, which optional exposure fields exist and how often they
    are filled, and the distribution of days between repeated administrations.
    tasks/disease_course/therapy.py derives depot coverage from those intervals.

Read-only; writes nothing.

Usage:
  python preprocessing/cli/discover_structured_codes.py
  python preprocessing/cli/discover_structured_codes.py --mrn-file /path/to/cohort.csv
"""

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl  # noqa: E402

from preprocessing.config import DEFAULT_PROSTATE_MRN_CSV, PROFILE_DATA_PATH  # noqa: E402
from preprocessing.notes import load_selected_mrns  # noqa: E402
from preprocessing.structured import (  # noqa: E402
    MEDICATION_OPTIONAL_COLUMNS,
    PSA_CODES,
    TESTOSTERONE_CODES,
    classify_medication,
    load_medications,
    source_path,
)

LAB_KEYWORDS = r"(?i)\b(?:psa|prostate|testost)"


def parse_args():
    parser = argparse.ArgumentParser(description="Report structured lab and medication codes.")
    parser.add_argument("--profile-data-path", type=Path, default=PROFILE_DATA_PATH)
    parser.add_argument("--mrns", default=None, help="Comma-separated MRNs.")
    parser.add_argument("--mrn-file", type=Path, default=DEFAULT_PROSTATE_MRN_CSV,
                        help="Cohort CSV; pass an empty string to scan every MRN.")
    parser.add_argument("--top", type=int, default=60)
    return parser.parse_args()


def _mrn_filter(lazy, mrns):
    if mrns is None:
        return lazy
    return lazy.filter(pl.col("DFCI_MRN").cast(pl.Int64, strict=False).is_in(sorted(mrns)))


def report_labs(path, mrns, top):
    lazy = _mrn_filter(pl.scan_parquet(path), mrns)
    columns = lazy.collect_schema().names()
    unit = pl.col("RESULT_UOM_NM") if "RESULT_UOM_NM" in columns else pl.lit(None)
    counts = (
        lazy.filter(
            pl.col("TEST_TYPE_CD").cast(pl.Utf8).str.to_uppercase().is_in(
                list(PSA_CODES + TESTOSTERONE_CODES)
            )
            | pl.col("TEST_TYPE_DESCR").cast(pl.Utf8).str.contains(LAB_KEYWORDS)
            | pl.col("TEST_TYPE_CD").cast(pl.Utf8).str.contains(LAB_KEYWORDS)
        )
        .group_by(
            pl.col("TEST_TYPE_CD").cast(pl.Utf8),
            pl.col("TEST_TYPE_DESCR").cast(pl.Utf8),
            unit.cast(pl.Utf8).alias("RESULT_UOM_NM"),
        )
        .agg(pl.len().alias("n"), pl.col("DFCI_MRN").n_unique().alias("patients"))
        .sort("n", descending=True)
        .collect()
    )
    print(f"\n=== LABS codes matching PSA / testosterone ({path}) ===")
    print(f"Configured PSA codes: {PSA_CODES}")
    print(f"Configured testosterone codes: {TESTOSTERONE_CODES}")
    with pl.Config(tbl_rows=top, fmt_str_lengths=60):
        print(counts.head(top))
    print(f"Optional lab identity columns present: "
          f"{[c for c in columns if 'SPECIMEN' in c or 'ACCESSION' in c or c.endswith('_ID')]}")


def report_medications(path, mrns, top):
    lazy = _mrn_filter(pl.scan_parquet(path), mrns)
    columns = lazy.collect_schema().names()
    names = (
        lazy.group_by(pl.col("NCI_PREFERRED_MED_NM").cast(pl.Utf8))
        .agg(pl.len().alias("n"), pl.col("DFCI_MRN").n_unique().alias("patients"))
        .sort("n", descending=True)
        .collect()
    )
    mapped = names.with_columns(
        pl.col("NCI_PREFERRED_MED_NM")
        .map_elements(lambda n: (classify_medication(n) or (None, None))[0], return_dtype=pl.Utf8)
        .alias("drug_class")
    )
    print(f"\n=== MEDICATIONS mapped to a drug class ({path}) ===")
    with pl.Config(tbl_rows=top, fmt_str_lengths=60):
        print(mapped.filter(pl.col("drug_class").is_not_null()).head(top))
        print("\nUnmapped names containing prostate-therapy stems (check for misses):")
        print(
            mapped.filter(
                pl.col("drug_class").is_null()
                & pl.col("NCI_PREFERRED_MED_NM").str.contains(
                    r"(?i)lutamide|relin|relix|abirat|taxel|radium|lutet|sipuleu|parib|platin"
                )
            ).head(top)
        )
    print("\nMedication columns present:", columns)
    print("Optional exposure fields resolved:", {
        name: next((c for c in candidates if c in columns), None)
        for name, candidates in MEDICATION_OPTIONAL_COLUMNS.items()
    })


def report_exposure(path, mrns):
    meds = load_medications(mrns, path)
    adt = meds.filter(pl.col("drug_class").is_in(["depot_adt", "oral_adt", "antiandrogen"]))
    print("\n=== ADT exposure-field fill rates (fraction non-null) ===")
    if adt.is_empty():
        print("No ADT records.")
        return
    fields = list(MEDICATION_OPTIONAL_COLUMNS)
    rates = adt.group_by("agent").agg(
        pl.len().alias("records"),
        *[pl.col(f).is_not_null().mean().round(3).alias(f) for f in fields],
    ).sort("records", descending=True)
    with pl.Config(tbl_cols=len(fields) + 2):
        print(rates)

    print("\n=== Days between consecutive administrations of the same depot agent ===")
    intervals = []
    for (_, agent), group in adt.filter(pl.col("drug_class") == "depot_adt").group_by(
        ["DFCI_MRN", "agent"]
    ):
        dates = sorted({date.fromisoformat(d) for d in group["date"].to_list()})
        intervals += [{"agent": agent, "days": (b - a).days} for a, b in zip(dates, dates[1:])]
    if not intervals:
        print("No repeated depot administrations.")
        return
    frame = pl.DataFrame(intervals)
    print(
        frame.group_by("agent").agg(
            pl.len().alias("intervals"),
            *[pl.col("days").quantile(q).alias(f"p{int(q * 100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
        ).sort("intervals", descending=True)
    )
    bins = [0, 14, 45, 75, 105, 135, 200, 270, 400]
    print("Interval histogram (days):")
    print(
        frame.with_columns(pl.col("days").cut(bins).alias("bin"))
        .group_by("bin").agg(pl.len().alias("n")).sort("bin")
    )


def main():
    args = parse_args()
    mrn_file = args.mrn_file if args.mrn_file and str(args.mrn_file) else None
    if mrn_file is not None and not Path(mrn_file).exists():
        print(f"Cohort file not found ({mrn_file}); scanning every MRN.")
        mrn_file = None
    mrns = load_selected_mrns(args.mrns, mrn_file)
    report_labs(source_path("labs", args.profile_data_path), mrns, args.top)
    meds_path = source_path("medications", args.profile_data_path)
    report_medications(meds_path, mrns, args.top)
    report_exposure(meds_path, mrns)


if __name__ == "__main__":
    main()
