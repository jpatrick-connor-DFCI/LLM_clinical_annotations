"""Build dated progression events from the disease-course extraction (no LLM calls).

Reads the course findings (tasks/disease_course/run_course_extraction.py), LABS
PSA, MEDICATIONS and the vital-status registry, applies
tasks/progression/events.py, and writes:

  progression_events.parquet       one row per event: type (psa | radiographic
                                   | clinical | unspecified | death), basis
                                   (statement | pcwg3 | registry), date, the
                                   therapy exposure on that date, PCWG3 audit
  progression_labels.parquet       one row per cohort patient: first date of
                                   each event type, registry death / last-alive
                                   dates and conflict flags
  progression_labels.meta.parquet  versions, course run, structured sources,
                                   rule defaults

PFS is chosen downstream against an index date; see
tasks.progression.events.first_event_after.

Usage:
  python tasks/progression/build_progression_events.py --output-dir /path/to/course_out
"""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.disease_course import builder  # noqa: E402
from tasks.progression.events import (  # noqa: E402
    PROGRESSION_EVENTS_VERSION,
    STATEMENT_TYPES,
    patient_events,
)
from tasks.onset import earliest  # noqa: E402

EVENT_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "event_type": pl.Utf8,
    "basis": pl.Utf8,
    "event_date": pl.Utf8,
    "date_precision": pl.Utf8,
    "source_ids": pl.List(pl.Utf8),
    "quote": pl.Utf8,
    "therapy_stated": pl.Utf8,
    "confirmation_date": pl.Utf8,
    "episode_agent": pl.Utf8,
    "episode_start": pl.Utf8,
    "episode_source": pl.Utf8,
    "pcwg3_branch": pl.Utf8,
    "baseline_id": pl.Utf8,
    "nadir_id": pl.Utf8,
    "psa_value": pl.Float64,
    "psa_threshold": pl.Float64,
    "therapies_active": pl.List(pl.Utf8),
    "therapies_possible": pl.List(pl.Utf8),
    "adt_active": pl.Boolean,
    "exposure_basis": pl.Utf8,
    "exposure_certainty": pl.Utf8,
}

LABEL_SCHEMA = {
    "DFCI_MRN": pl.Int64,
    "review_status": pl.Utf8,
    "death_date": pl.Utf8,
    "death_date_conflict": pl.Boolean,
    "death_date_candidates": pl.List(pl.Utf8),
    "last_alive_date": pl.Utf8,
    "last_alive_after_death": pl.Boolean,
}
for _type in STATEMENT_TYPES + ("any",):
    LABEL_SCHEMA.update({
        f"first_{_type}_progression_date": pl.Utf8,
        f"first_{_type}_progression_precision": pl.Utf8,
        f"first_{_type}_progression_basis": pl.Utf8,
    })
LABEL_SCHEMA.update({
    "n_progression_events": pl.Int64,
    "pcwg3_unevaluable": pl.List(pl.Utf8),
    "gate_rejections": pl.List(pl.Utf8),
})
REGISTRY_COLUMNS = ("death_date", "death_date_conflict", "death_date_candidates",
                    "last_alive_date", "last_alive_after_death")


def _first(events, types):
    rows = [
        {**e, "source_note_date": None} for e in events
        if e["event_type"] in types
    ]
    return earliest(rows)


def build(inputs, structured):
    event_rows, label_rows = [], []
    for mrn in sorted(inputs["status"]):
        status = inputs["status"][mrn]
        reviewed = status not in builder.UNLABELED_STATUSES
        vital = (structured["vital_status"].get(mrn) or [None])[0]
        events, reasons = patient_events(
            inputs["usable"].get(mrn, []),
            structured["psa"].get(mrn, []),
            structured["medications"].get(mrn, []),
            vital,
            notes_reviewed=reviewed,
        )
        event_rows += [{"DFCI_MRN": mrn, **e} for e in events]
        progression = [e for e in events if e["event_type"] != "death"]
        fields = {}
        for event_type in STATEMENT_TYPES + ("any",):
            types = STATEMENT_TYPES if event_type == "any" else (event_type,)
            first = _first(progression, types)
            fields.update({
                f"first_{event_type}_progression_date": first and first["event_date"],
                f"first_{event_type}_progression_precision": first and first["date_precision"],
                f"first_{event_type}_progression_basis": first and first["basis"],
            })
        fields["n_progression_events"] = len(progression)
        fields["pcwg3_unevaluable"] = reasons
        fields = builder.blank_unreviewed(fields, status)
        label_rows.append({
            "DFCI_MRN": mrn,
            "review_status": status,
            **{c: (vital or {}).get(c) for c in REGISTRY_COLUMNS},
            **fields,
            "gate_rejections": inputs["rejected"].get(mrn, []),
        })
    return event_rows, label_rows


def main(argv=None):
    args = builder.build_parser(
        "Build progression events from disease-course findings."
    ).parse_args(argv)
    inputs = builder.load_course(args)
    structured, fingerprint = builder.load_structured(
        ["psa", "medications", "vital_status"], inputs["cohort"], args.profile_data_path
    )
    event_rows, label_rows = build(inputs, structured)
    metadata = builder.base_metadata(
        inputs, fingerprint, progression_events_version=PROGRESSION_EVENTS_VERSION
    )
    events = builder.write_output(
        args.output_dir / "progression_events.parquet", event_rows, EVENT_SCHEMA, metadata
    )
    labels = builder.write_output(
        args.output_dir / "progression_labels.parquet", label_rows, LABEL_SCHEMA, metadata
    )
    print(f"Wrote {events.height} progression events and labels for {labels.height} patients "
          f"to {args.output_dir}")
    print(events.group_by("event_type", "basis").len().sort("event_type", "basis"))
    return events, labels


if __name__ == "__main__":
    main()
