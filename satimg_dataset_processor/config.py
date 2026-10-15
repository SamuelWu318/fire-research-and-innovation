"""Global settings for the TS-SatFire prediction dataset pipeline.

Edit these constants to change the default behavior of dataset_gen_pred.py,
satimg_dataset_processor.hf_dataset, and satimg_dataset_processor.tessera.
Every function also accepts the matching argument, so a notebook can override
a setting for one call without editing this file.
"""

# ----- Window generation ----- #

# Input days per window. A fire needs TS_LENGTH + 1 VIIRS_Day images (the
# extra day is the target), so fires with fewer images produce no windows.
# The TS-SatFire paper's prediction benchmark uses 6 input days.
TS_LENGTH = 10

# Days between the first input day of consecutive windows.
TS_INTERVAL = 3

# ----- Hugging Face dataset repositories ----- #

SRC_REPO = "SamuelWu318/ts-satfire"                    # raw GeoTIFFs + ROI CSVs
PROCESSED_REPO = "SamuelWu318/ts-satfire-processed"    # old-format NPZ data.zip
TESSERA_REPO = "SamuelWu318/ts-tesserafire"            # prior-year TESSERA per fire

# Windowed NPZs live in one repo per (TS_LENGTH, TS_INTERVAL): file names do
# not encode the window length, so different settings must never share a repo.
# The original 10-day, 3-day-stride windows keep the unsuffixed name.
WINDOW_REPO_BASE = "SamuelWu318/ts-satfire-processed-window"
LEGACY_WINDOW_SETTINGS = (10, 3)


def window_repo_for(ts_length=TS_LENGTH, ts_interval=TS_INTERVAL):
    """Window repo holding windows of `ts_length` days every `ts_interval` days."""
    if (ts_length, ts_interval) == LEGACY_WINDOW_SETTINGS:
        return WINDOW_REPO_BASE
    return f"{WINDOW_REPO_BASE}-ts{ts_length}-it{ts_interval}"


WINDOW_REPO = window_repo_for()

# Written at the root of each window repo by process_data; checked before
# new windows are added so one repo never mixes window settings.
WINDOW_CONFIG_FILE = "window_config.json"

# ----- Prediction labels ----- #

# The target is the pixels newly burned on the day after the input window:
# "the difference between the burned area masks of the last observed day and
# the next day" (TS-SatFire paper). label_sel picks the burned-area mask:
#   0: union of accumulated VIIRS active fire (AF) and the NIFC perimeters
#      (BA band). The paper's training label, and test option (a).
#   1: accumulated AF only. Test option (b), chosen per fire by visual
#      inspection "to avoid the high commission errors of NIFC perimeters".
# Train/val fires use TRAIN_VAL_LABEL_SEL; test fires use the label_sel
# column of the 2021 ROI CSV, which records the paper's per-fire choice.
# The released TS-SatFire code used 1 for train/val; set 1 to reproduce it.
TRAIN_VAL_LABEL_SEL = 0

# Window repo manifest {"<split>/p<id>.npz": label_sel}. Archives missing from
# it were built before it existed, with LEGACY_LABEL_SEL.
LABEL_SEL_FILE = "label_sel.json"
LEGACY_LABEL_SEL = 1

# ----- Train / validation / test split ----- #

TRAIN_VAL_YEARS = ('2017', '2018', '2019', '2020')
TEST_YEARS = ('2021',)
VAL_IDS = [
    '20568194', '20701026', '20562846', '20700973', '24462610',
    '24462788', '24462753', '24103571', '21998313', '21751303',
    '22141596', '21999381', '23301962', '22712904', '22713339',
]
