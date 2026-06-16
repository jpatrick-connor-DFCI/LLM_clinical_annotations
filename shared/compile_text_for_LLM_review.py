import os
import numpy as np
import pandas as pd

# Note window (days) kept around the platinum start date for LLM review.
NOTE_WINDOW_DAYS = 90

# Column name is intentionally misspelled to match the source annotation TSV.
INDICATION_COL = 'Inidcation of Platinum Therapy'

DATA_PATH = os.environ.get(
    'LLM_ANNOTATIONS_DATA_PATH',
    os.environ.get('CAIA_COMPASS_DATA_PATH', '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'),
)
baca_df = pd.read_csv(
    os.path.join(DATA_PATH, 'baca_lab_patient_annotations.tsv'),
    sep="\t",
    encoding="utf-8",
    encoding_errors="replace")
text_df = pd.read_csv(os.path.join(DATA_PATH, 'prostate_text_data.csv'))

if INDICATION_COL not in baca_df.columns:
    raise KeyError(
        f"Expected column {INDICATION_COL!r} in baca_lab_patient_annotations.tsv; "
        f"found {list(baca_df.columns)}. If the source header spelling changed, update INDICATION_COL."
    )

candidate_patients = baca_df.loc[baca_df[INDICATION_COL].isna(),
                                 ['DFCI_MRN', 'PLATINUM_CHEMO_MED', 'MEDICATION_START_TIME']]
LLM_text_df = text_df.merge(candidate_patients, on='DFCI_MRN', how='inner')
event_dt = pd.to_datetime(LLM_text_df['EVENT_DATE'], errors='coerce', utc=True)
med_dt = pd.to_datetime(LLM_text_df['MEDICATION_START_TIME'], errors='coerce', utc=True)

LLM_text_df['NOTE_DAYS_REL_PLATINUM'] = (event_dt - med_dt).dt.days
candidate_LLM_text_df = (LLM_text_df
    .loc[np.abs(LLM_text_df['NOTE_DAYS_REL_PLATINUM']) <= NOTE_WINDOW_DAYS,
         ['EVENT_DATE', 'DFCI_MRN', 'NOTE_TYPE', 'CLINICAL_TEXT']]
    .sort_values(by=['DFCI_MRN', 'EVENT_DATE']))
candidate_LLM_text_df.to_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'), index=False)
