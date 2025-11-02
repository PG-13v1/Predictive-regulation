import os
from utils.data_utils import load_secom, basic_impute_and_scale
from prophet_forecast import run_prophet_forecasts
from lstm_autoencoder import train_lstm_autoencoder
from utils.train_utils import set_seed

# ==========================
# 1. GLOBAL CONFIG
# ==========================
SEED = 42
DATA_PATH = "data/secom_combined.csv"
MODEL_PATH = "models/lstm_autoencoder.h5"
FORECAST_DIR = "outputs/forecasts"



def main():
    print("\n🚀 Starting SECOM Predictive Maintenance Pipeline...\n")
    set_seed(SEED)

    # --- Step 1: Load and preprocess data ---
    print("📦 Loading and preprocessing data...")
    df = load_secom(DATA_PATH)
    df_clean = basic_impute_and_scale(df)

    # --- Step 2: Prophet forecasting ---
    print("\n🔮 Running Prophet forecasts...")
    os.makedirs(FORECAST_DIR, exist_ok=True)
    run_prophet_forecasts(df_clean, n_features=3, save_dir=FORECAST_DIR)

    # --- Step 3: LSTM Autoencoder training ---
    print("\n🤖 Training LSTM Autoencoder...")
    train_lstm_autoencoder(df_clean, save_path=MODEL_PATH)

    print("\n✅ All steps completed successfully!")
    print(f"   LSTM model saved at: {MODEL_PATH}")
    print(f"   Forecast plots saved in: {FORECAST_DIR}\n")


# ==========================
# 3. ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()