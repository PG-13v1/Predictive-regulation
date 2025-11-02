from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42


# LSTM params
LSTM_SEQ_LEN = 1 # SECOM is per-run; we treat features as "sequence" dimension; seq_len 1 for standard AE
LSTM_LATENT = 64
BATCH_SIZE = 32
EPOCHS = 100


# Prophet params
PROPHET_PERIODS = 10


# Preprocessing
VAR_THRESH = 1e-5
MIN_FEATURES = 10