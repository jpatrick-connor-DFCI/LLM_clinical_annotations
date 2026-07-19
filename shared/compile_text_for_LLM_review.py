import os

import polars as pl

# Note window (days) kept around the platinum start date for LLM review.
NOTE_WINDOW_DAYS = 90

# Column name is intentionally misspelled to match the source annotation TSV.
INDICATION_COL = 'Inidcation of Platinum Therapy'

DATA_PATH = os.environ.get(
    'LLM_ANNOTATIONS_DATA_PATH',
    os.environ.get('CAIA_COMPASS_DATA_PATH', '/data/gusev/USERS/jpconnor/data/LLM_annotations/'),
)
baca_df = pl.read_csv(
    os.path.join(DATA_PATH, 'baca_lab_patient_annotations.tsv'),
    separator="\t",
    encoding="utf8-lossy",
)
text_df = pl.scan_csv(os.path.join(DATA_PATH, 'prostate_text_data.csv'))

if INDICATION_COL not in baca_df.columns:
    raise KeyError(
        f"Expected column {INDICATION_COL!r} in baca_lab_patient_annotations.tsv; "
        f"found {baca_df.columns}. If the source header spelling changed, update INDICATION_COL."
    )

candidate_patients = baca_df.filter(pl.col(INDICATION_COL).is_null()).select(
    ['DFCI_MRN', 'PLATINUM_CHEMO_MED', 'MEDICATION_START_TIME']
)
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
candidate_LLM_text_df.write_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
