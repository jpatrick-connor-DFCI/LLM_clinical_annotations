import os
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.config import DEFAULT_PROFILE_NOTE_PATHS  # noqa: E402
from preprocessing.notes import load_profile_notes  # noqa: E402

# Note window (days) kept around the platinum start date for LLM review.
NOTE_WINDOW_DAYS = 90

# Column name is intentionally misspelled to match the source annotation Parquet.
INDICATION_COL = 'Inidcation of Platinum Therapy'

DATA_PATH = os.environ.get(
    'LLM_ANNOTATIONS_DATA_PATH',
    os.environ.get('CAIA_COMPASS_DATA_PATH', '/data/gusev/USERS/jpconnor/data/LLM_annotations/'),
)
baca_path = os.path.join(DATA_PATH, 'baca_lab_patient_annotations.parquet')
baca_df = pl.read_parquet(baca_path)
if INDICATION_COL not in baca_df.columns:
    raise KeyError(
        f"Expected column {INDICATION_COL!r} in {baca_path}; "
        f"found {baca_df.columns}. If the source header spelling changed, update INDICATION_COL."
    )

candidate_patients = baca_df.filter(pl.col(INDICATION_COL).is_null()).select(
    ['DFCI_MRN', 'PLATINUM_CHEMO_MED', 'MEDICATION_START_TIME']
)
selected_mrns = set(
    candidate_patients['DFCI_MRN'].cast(pl.Int64, strict=False).drop_nulls().to_list()
)
text_df = load_profile_notes(DEFAULT_PROFILE_NOTE_PATHS, selected_mrns).lazy()
candidate_LLM_text_df = (
    text_df.join(candidate_patients.lazy(), on='DFCI_MRN', how='inner')
    .with_columns(
        event_dt=pl.col('EVENT_DATE').str.to_datetime(strict=False, time_unit='us'),
        med_dt=pl.col('MEDICATION_START_TIME').str.to_datetime(strict=False, time_unit='us'),
    )
    .with_columns(
        NOTE_DAYS_REL_PLATINUM=(pl.col('event_dt') - pl.col('med_dt')).dt.total_days()
    )
    .filter(pl.col('NOTE_DAYS_REL_PLATINUM').abs() <= NOTE_WINDOW_DAYS)
    .select(['EVENT_DATE', 'DFCI_MRN', 'NOTE_TYPE', 'CLINICAL_TEXT'])
    .sort(['DFCI_MRN', 'EVENT_DATE'])
    .collect()
)
candidate_LLM_text_df.write_parquet(
    os.path.join(DATA_PATH, 'LLM_candidate_text_data.parquet'),
    compression='zstd',
)
