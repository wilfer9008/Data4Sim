from __future__ import annotations
from pathlib import Path

# -------- Data --------
DATA_DIR = Path("/home/mkfari/Data4Sim_New/final data4sim/data_preprocessed")
CSV_GLOB = "S*.csv"
INPUT_COL = "input"
OUTPUT_COL = "output"

Freq = 100


SMALL_WIN = 200
STRIDE = 50

# Big window = sequence of small windows
MIN_SEQ = 4
MAX_SEQ = 24
RAND_LEN_TRAIN = True  # random big-window length only in training

# Split by subject (recommended to avoid leakage)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# -------- Training --------
BATCH = 8          # shuffles big windows only when shuffle=True
EPOCHS = 30
LR = 1e-3
WD = 1e-5
CLIP = 1.0
DEVICE = "cuda"
OUT_DIR = Path("runs/lstm_process_prediction")

# -------- Model --------
TOKEN_EMB = 64     # embedding size for categorical 'input' tokens
INNER_HID = 128    # encodes each small window
OUTER_HID = 256    # encodes sequence of small windows
LAYERS = 2
DROPOUT = 0.2
