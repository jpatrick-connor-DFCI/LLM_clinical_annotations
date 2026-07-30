"""Read-only diagnostic for the longitudinal_NEPC date-reproducibility bug (Phase 2).

Reads avpc_nepc_extractions_raw.tsv / avpc_nepc_timeline.tsv from an existing
--output-dir and reports statistics needed to decide whether the to_iso_date /
resolve_date reproducibility bug (fixed by parse_stated_date in
preprocessing/longitudinal.py) actually mattered in practice. Writes nothing;
run this before deciding whether any date-parsing fix is worth carrying further.

Usage:
  python tasks/longitudinal_NEPC/diagnose_dates.py --output-dir /path/to/output
"""

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_DATA_PATH  # noqa: E402
from preprocessing.notes import to_iso_date  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_avpc_nepc_timeline"

_FULL_ISO_RE_SOURCE = r"^\d{4}-\d{2}-\d{2}$"
_SANITY_MIN_YEAR = 1990


def _read_tsv(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pl.read_csv(path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)


def blast_radius(raw, date_col):
    """Fraction of raw stated-date strings that are NOT already full YYYY-MM-DD.

    This is the headline number: it measures the bug directly from data on disk,
    independent of any parser fix, by checking the raw text the LLM actually
    returned before either to_iso_date or parse_stated_date touch it.
    """
    values = [v for v in raw[date_col].to_list() if v not in (None, "", "null", "None")]
    if not values:
        return {"n": 0, "n_not_full_iso": 0, "frac_not_full_iso": None}
    import re

    full_iso = re.compile(_FULL_ISO_RE_SOURCE)
    n_not_full = sum(1 for v in values if not full_iso.match(str(v).strip()))
    return {
        "n": len(values),
        "n_not_full_iso": n_not_full,
        "frac_not_full_iso": n_not_full / len(values),
    }


def date_source_distribution(timeline, criterion_col):
    """date_source counts overall and per criterion, plus date_precision if present."""
    overall = Counter(timeline["date_source"].to_list())
    has_precision = "date_precision" in timeline.columns
    precision_overall = Counter(timeline["date_precision"].to_list()) if has_precision else None

    per_criterion = {}
    if criterion_col in timeline.columns:
        for crit in sorted(set(timeline[criterion_col].to_list())):
            sub = timeline.filter(pl.col(criterion_col) == crit)
            per_criterion[crit] = Counter(sub["date_source"].to_list())

    return {
        "overall": dict(overall),
        "precision_overall": dict(precision_overall) if precision_overall else None,
        "per_criterion": {k: dict(v) for k, v in per_criterion.items()},
    }


def null_event_date_rate(timeline, criterion_col, date_col):
    """Null event_date count/rate per criterion."""
    if criterion_col not in timeline.columns:
        return {}
    out = {}
    for crit in sorted(set(timeline[criterion_col].to_list())):
        sub = timeline.filter(pl.col(criterion_col) == crit)
        vals = sub[date_col].to_list()
        n = len(vals)
        n_null = sum(1 for v in vals if v in (None, "", "null", "None"))
        out[crit] = {"n": n, "n_null": n_null, "rate_null": (n_null / n) if n else None}
    return out


def diagnosis_minus_note_date(raw, diagnosis_col, note_date_col):
    """Distribution of (diagnosis_date - source_note_date) in days.

    Negative is expected (a note recites history predating itself). Large positive
    is a bug signature: a diagnosis dated after the note that documents it.
    """
    deltas = []
    for d_raw, n_raw in zip(raw[diagnosis_col].to_list(), raw[note_date_col].to_list()):
        d_iso = to_iso_date(d_raw)
        n_iso = to_iso_date(n_raw)
        if d_iso is None or n_iso is None:
            continue
        delta = (date.fromisoformat(d_iso) - date.fromisoformat(n_iso)).days
        deltas.append(delta)
    if not deltas:
        return {"n": 0}
    deltas.sort()
    n = len(deltas)
    large_positive = sum(1 for d in deltas if d > 0)
    return {
        "n": n,
        "min": deltas[0],
        "p50": deltas[n // 2],
        "max": deltas[-1],
        "n_positive": large_positive,
        "frac_positive": large_positive / n,
    }


def sanity_bounds(raw, diagnosis_col, min_year=_SANITY_MIN_YEAR):
    """Count stated dates outside [min_year-01-01, today] -- implausible values."""
    today = date.today()
    n_pre_min = 0
    n_post_today = 0
    n_checked = 0
    for v in raw[diagnosis_col].to_list():
        iso = to_iso_date(v)
        if iso is None:
            continue
        n_checked += 1
        d = date.fromisoformat(iso)
        if d.year < min_year:
            n_pre_min += 1
        if d > today:
            n_post_today += 1
    return {"n_checked": n_checked, "n_pre_min_year": n_pre_min, "n_post_today": n_post_today, "min_year": min_year}


def copy_forward_fraction(timeline):
    """Fraction of timeline rows relying on the note_date fallback (date_source == 'note_date').

    dedupe_candidates (preprocessing/longitudinal.py) keeps the EARLIEST note_date
    per (mrn, snippet); when date_source is 'note_date', source_note_date is when the
    TEXT first appeared, which may predate the actual event for a copy-forward note.
    """
    if "date_source" not in timeline.columns or timeline.height == 0:
        return {"n": 0}
    n = timeline.height
    n_note_date = timeline.filter(pl.col("date_source") == "note_date").height
    return {"n": n, "n_note_date_fallback": n_note_date, "frac_note_date_fallback": n_note_date / n}


def run(args):
    raw = _read_tsv(args.output_dir / "avpc_nepc_extractions_raw.tsv")
    timeline = _read_tsv(args.output_dir / "avpc_nepc_timeline.tsv")

    if raw is None and timeline is None:
        print(f"No avpc_nepc_extractions_raw.tsv / avpc_nepc_timeline.tsv found under {args.output_dir}")
        print("Nothing to diagnose against real data. Run against a synthetic fixture instead,")
        print("or point --output-dir at a real pipeline run's output directory.")
        return

    print("=" * 70)
    print(f"diagnose_dates.py — {args.output_dir}")
    print("=" * 70)

    if raw is not None:
        print("\n[1] Blast radius — fraction of raw diagnosis_date NOT already full YYYY-MM-DD")
        result = blast_radius(raw, "diagnosis_date")
        if result["frac_not_full_iso"] is None:
            print("    no non-null diagnosis_date values found")
        else:
            print(
                f"    n={result['n']}  not-full-iso={result['n_not_full_iso']}"
                f"  ({result['frac_not_full_iso']:.1%})"
            )
    else:
        print("\n[1] Blast radius — SKIPPED (avpc_nepc_extractions_raw.tsv not found)")

    if timeline is not None:
        print("\n[2] date_source / date_precision distribution")
        dist = date_source_distribution(timeline, "criterion_added")
        print(f"    overall date_source: {dist['overall']}")
        if dist["precision_overall"] is not None:
            print(f"    overall date_precision: {dist['precision_overall']}")
        else:
            print("    date_precision column not present (pre-Phase-2 timeline)")
        print("    per criterion:")
        for crit, counts in dist["per_criterion"].items():
            print(f"      {crit}: {counts}")

        print("\n[3] Null event_date rate per criterion")
        null_rates = null_event_date_rate(timeline, "criterion_added", "event_date")
        for crit, stats in null_rates.items():
            rate = stats["rate_null"]
            rate_str = f"{rate:.1%}" if rate is not None else "n/a"
            print(f"      {crit}: n={stats['n']} n_null={stats['n_null']} rate={rate_str}")
    else:
        print("\n[2]/[3] SKIPPED (avpc_nepc_timeline.tsv not found)")

    if raw is not None:
        print("\n[4] diagnosis_date - source_note_date (days)")
        delta_stats = diagnosis_minus_note_date(raw, "diagnosis_date", "source_note_date")
        if delta_stats["n"] == 0:
            print("    no comparable pairs found")
        else:
            print(
                f"    n={delta_stats['n']}  min={delta_stats['min']}  p50={delta_stats['p50']}"
                f"  max={delta_stats['max']}"
            )
            print(
                f"    positive (diagnosis AFTER note -- possible bug): "
                f"{delta_stats['n_positive']} ({delta_stats['frac_positive']:.1%})"
            )

        print("\n[5] Absolute sanity bounds")
        bounds = sanity_bounds(raw, "diagnosis_date")
        print(
            f"    n_checked={bounds['n_checked']}  pre-{bounds['min_year']}={bounds['n_pre_min_year']}"
            f"  post-today={bounds['n_post_today']}"
        )
    else:
        print("\n[4]/[5] SKIPPED (avpc_nepc_extractions_raw.tsv not found)")

    if timeline is not None:
        print("\n[6] Copy-forward note_date fallback fraction")
        cf = copy_forward_fraction(timeline)
        if cf["n"] == 0:
            print("    no timeline rows / no date_source column")
        else:
            print(
                f"    n={cf['n']}  note_date_fallback={cf['n_note_date_fallback']}"
                f"  ({cf['frac_note_date_fallback']:.1%})"
            )
    else:
        print("\n[6] SKIPPED (avpc_nepc_timeline.tsv not found)")

    print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Directory containing avpc_nepc_extractions_raw.tsv / avpc_nepc_timeline.tsv")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
