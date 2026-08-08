"""Atomic helpers for pipeline-owned Parquet artifacts.

Clinical annotation artifacts are intentionally Parquet-only.  These helpers
centralize atomic replacement and the small append/upsert operations needed by
the resumable runners so every persisted artifact follows one storage contract.
"""

from pathlib import Path

import polars as pl


PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 10


def rows_to_frame(rows, columns):
    """Build a frame with a deterministic column order from row dictionaries."""
    rows = list(rows)
    if not rows:
        return pl.DataFrame(
            {column: pl.Series(column, [], dtype=pl.String) for column in columns}
        )
    return pl.DataFrame(
        {column: [row.get(column) for row in rows] for column in columns},
        strict=False,
    )


def write_parquet_atomic(frame, path):
    """Write one Parquet file by atomic same-directory replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.write_parquet(
        temporary,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )
    temporary.replace(path)


def write_rows_atomic(path, rows, columns):
    write_parquet_atomic(rows_to_frame(rows, columns), path)


def append_rows_atomic(path, rows, columns):
    """Append rows by atomically replacing a single Parquet artifact.

    Callers should pass a batch (normally one completed patient or larger), not
    individual fields.  ``diagonal_relaxed`` safely promotes columns that were
    null-only in an earlier batch.
    """
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    incoming = rows_to_frame(rows, columns)
    if path.exists() and path.stat().st_size:
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, incoming], how="diagonal_relaxed")
    else:
        combined = incoming
    write_parquet_atomic(combined.select(columns), path)


def write_metadata(path, payload):
    """Persist a single metadata record as Parquet."""
    write_parquet_atomic(pl.DataFrame([payload], strict=False), path)


def read_metadata(path):
    """Read a single-record Parquet metadata artifact, or return ``None``."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return None
    try:
        frame = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError):
        return None
    if frame.is_empty():
        return None
    return frame.row(0, named=True)
