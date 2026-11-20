"""Merge the per-note subtype labels into one primary-subtype table.

No LLM calls. Reads `nepc_labels`, `avpc_labels` and `avpc_adjacent_labels`
(full join on DFCI_MRN) plus optional `arlpc_labels` (left join, annotation
only), and adds:

  - `primary_label`: nepc > avpc > avpc_adjacent > prad, keyed on the broad
    `nepc_status` (any gated NE evidence, including t3 and IHC-only).
  - `primary_label_definitive`: the same cascade keyed on `nepc_t1`
    (pathology-defined definitive NEPC). Broad-only NE patients fall through
    to the AVPC / adjacent / prad branches.
  - `small_cell_discordant`: AVPC C1 (or adjacent C1) says small-cell
    histology but the NEPC pipeline did not label the patient nepc. The two
    pipelines read different trigger notes with different prompts, so this
    flags patients for manual review; it changes no label.

Each label is null when a higher-precedence input is unresolved (failed,
not_run or partial review, or the patient is missing from one file).

Every input's `<prefix>_labels.meta.parquet` must record the gate and
criteria versions the current code would write: stale labels raise with a
`--rebuild-labels-only` hint instead of merging silently. The merged output
gets a `.meta.parquet` recording each input's path, sha256 and label meta.
"""

import json
from pathlib import Path

import polars as pl

from preprocessing.longitudinal import file_sha256
from preprocessing.parquet_io import read_metadata, write_metadata, write_parquet_atomic
from tasks.arlpc import run_arlpc_extraction as arlpc_runner
from tasks.avpc import run_avpc_extraction as avpc_runner
from tasks.avpc_adjacent import run_avpc_adjacent_extraction as adjacent_runner
from tasks.nepc import run_nepc_extraction as nepc_runner
from tasks.trial_guard import TRIAL_GUARD_VERSION

# v2: adds primary_label_definitive, small_cell_discordant, the input version
# check and the meta sidecar.
PRIMARY_SUBTYPE_VERSION = "primary-subtype-v2"

# Columns both NEPC and AVPC write; prefixed so the join keeps both.
SHARED_COLUMNS = ("review_status", "gate_rejections")

PIPELINES = {
    "nepc": (nepc_runner.SPEC, "nepc.ipynb / tasks/nepc/run_nepc_extraction.py"),
    "avpc": (avpc_runner.SPEC, "avpc.ipynb / tasks/avpc/run_avpc_extraction.py"),
    "avpc_adjacent": (
        adjacent_runner.SPEC,
        "avpc_adjacent.ipynb / tasks/avpc_adjacent/run_avpc_adjacent_extraction.py",
    ),
    "arlpc": (arlpc_runner.SPEC, "arlpc.ipynb / tasks/arlpc/run_arlpc_extraction.py"),
}
REQUIRED = ("nepc", "avpc", "avpc_adjacent")


def labels_meta_path(labels_path):
    labels_path = Path(labels_path)
    return labels_path.with_name(f"{labels_path.stem}.meta.parquet")


def expected_label_versions(spec):
    """The version fields `note_extraction.write_labels` would record now."""
    return {"trial_guard_version": TRIAL_GUARD_VERSION,
            **{k: str(v) for k, v in spec.label_versions.items()}}


def check_label_versions(name, labels_path):
    """Return the input's label meta; raise if it is missing or stale."""
    spec, runner = PIPELINES[name]
    meta_path = labels_meta_path(labels_path)
    meta = read_metadata(meta_path)
    hint = f"re-run {runner} with --rebuild-labels-only (no LLM calls)"
    if meta is None:
        raise FileNotFoundError(f"{name}: no label metadata at {meta_path}; {hint}")
    stale = {
        key: {"labels": meta.get(key), "code": value}
        for key, value in expected_label_versions(spec).items()
        if meta.get(key) != value
    }
    if stale:
        raise ValueError(f"{name} labels at {labels_path} are stale {stale}; {hint}")
    return meta


def read_labels(name, path, check_versions=True):
    """Read one pipeline's labels, prefixing the shared columns."""
    meta = check_label_versions(name, path) if check_versions else read_metadata(labels_meta_path(path))
    frame = pl.read_parquet(path)
    if frame["DFCI_MRN"].is_duplicated().any():
        raise ValueError(f"{name} labels have duplicate DFCI_MRN rows")
    if name in {"nepc", "avpc"}:
        frame = frame.rename({c: f"{name}_{c}" for c in SHARED_COLUMNS if c in frame.columns})
    return frame, meta


def _cascade(nepc_positive, nepc_negative):
    # Null comparisons are null, which `when` treats as false: unresolved
    # inputs fall through to the null otherwise.
    is_avpc, is_adjacent = pl.col("is_avpc"), pl.col("is_avpc_adjacent")
    return (
        pl.when(nepc_positive).then(pl.lit("nepc"))
        .when(nepc_negative & is_avpc).then(pl.lit("avpc"))
        .when(nepc_negative & is_avpc.not_() & is_adjacent).then(pl.lit("avpc_adjacent"))
        .when(nepc_negative & is_avpc.not_() & is_adjacent.not_()).then(pl.lit("prad"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


PRIMARY_LABEL = _cascade(pl.col("nepc_status") == "nepc", pl.col("nepc_status") == "non_nepc")
PRIMARY_LABEL_DEFINITIVE = _cascade(pl.col("nepc_t1") == True, pl.col("nepc_t1") == False)  # noqa: E712
SMALL_CELL_DISCORDANT = (
    ((pl.col("c1_met") == True) | (pl.col("avpc_adj_c1_met") == True))  # noqa: E712
    & (pl.col("nepc_status").fill_null("") != "nepc")
).fill_null(False)

LEADING = ("DFCI_MRN", "primary_label", "primary_label_definitive", "small_cell_discordant")


def merge_labels(nepc, avpc, adjacent, arlpc=None):
    """Full-join the three required tables, add the label columns, left-join ARLPC."""
    frames = (("nepc", nepc), ("avpc", avpc), ("avpc_adjacent", adjacent))
    for i, (name_a, a) in enumerate(frames):
        for name_b, b in frames[i + 1:]:
            overlap = (set(a.columns) & set(b.columns)) - {"DFCI_MRN"}
            if overlap:
                raise ValueError(f"Unexpected shared columns between {name_a} and {name_b}: "
                                 f"{sorted(overlap)}")
    merged = (
        nepc.join(avpc, on="DFCI_MRN", how="full", coalesce=True)
        .join(adjacent, on="DFCI_MRN", how="full", coalesce=True)
        .with_columns(
            PRIMARY_LABEL.alias("primary_label"),
            PRIMARY_LABEL_DEFINITIVE.alias("primary_label_definitive"),
            SMALL_CELL_DISCORDANT.alias("small_cell_discordant"),
        )
        .select(*LEADING, pl.exclude(*LEADING))
        .sort("DFCI_MRN")
    )
    if arlpc is not None:
        if arlpc["DFCI_MRN"].is_duplicated().any():
            raise ValueError("arlpc labels have duplicate DFCI_MRN rows")
        clash = (set(arlpc.columns) & set(merged.columns)) - {"DFCI_MRN"}
        if clash:
            raise ValueError(f"Unexpected shared columns with the ARLPC labels: {sorted(clash)}")
        # Annotation only: a left join never adds patients or changes a label.
        merged = merged.join(arlpc, on="DFCI_MRN", how="left")
    return merged


def small_cell_discordance(merged):
    """Patients whose AVPC C1 small-cell quote the NEPC labels do not reflect."""
    return merged.filter(pl.col("small_cell_discordant")).select(
        "DFCI_MRN", "nepc_status", "nepc_review_status", "nepc_gate_rejections",
        "c1_met", "c1_quote", "avpc_adj_c1_met", "avpc_adj_c1_quote",
    )


def build(paths, check_versions=True):
    """Read and merge. `paths` maps pipeline name -> labels path; arlpc optional.

    Returns ``(merged, inputs)``; `inputs` maps name -> {path, sha256, meta}.
    """
    missing = [name for name in REQUIRED if not Path(paths.get(name, "")).is_file()]
    if missing:
        raise FileNotFoundError(f"Run the {', '.join(missing)} pipeline(s) first; missing labels")
    frames, inputs = {}, {}
    for name in PIPELINES:
        path = paths.get(name)
        if path is None or not Path(path).is_file():
            continue
        frames[name], meta = read_labels(name, path, check_versions)
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
                        "meta": meta}
    merged = merge_labels(frames["nepc"], frames["avpc"], frames["avpc_adjacent"],
                          frames.get("arlpc"))
    return merged, inputs


def write_outputs(merged, inputs, output_path):
    """Write the merged labels and a meta sidecar recording their provenance."""
    output_path = Path(output_path)
    write_parquet_atomic(merged, output_path)
    payload = {"primary_subtype_version": PRIMARY_SUBTYPE_VERSION, "n_patients": merged.height}
    for name, info in inputs.items():
        payload[f"{name}_labels_path"] = info["path"]
        payload[f"{name}_labels_sha256"] = info["sha256"]
        payload[f"{name}_labels_meta"] = json.dumps(info["meta"], default=str, sort_keys=True)
    meta_path = labels_meta_path(output_path)
    write_metadata(meta_path, payload)
    return meta_path
