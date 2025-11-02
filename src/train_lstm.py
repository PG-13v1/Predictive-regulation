from sklearn.model_selection import train_test_split
import numpy as np
from models.lstm_autoencoder import build_lstm_autoencoder
from utils.data_utils import load_secom, basic_impute_and_scale
from utils.train_utils import set_seed
from utils.evaluation import evaluate_reconstruction

def run(out_path='models/lstm_autoencoder.h5'):
    # 1. Ensure reproducibility
    set_seed()

    # 2. Load and preprocess SECOM dataset
    df = load_secom()
    Xs = basic_impute_and_scale(df)

    # Separate target and features
    y = Xs['target'].values
    X = Xs.drop(columns=['target']).values

    # 3. Train-test split with stratification to preserve fail distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Reshape for LSTM input (samples, timesteps, features)
    #    Here we treat each observation as a single timestep.
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # 5. Build the LSTM Autoencoder
    model = build_lstm_autoencoder(input_dim=X_train.shape[2])

    # 6. Train the model
    model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )

    # 7. Save the trained model
    model.save(out_path)
    print(f"✅ Model saved at {out_path}")

    # 8. Evaluate reconstruction error to detect anomalies
    evaluate_reconstruction(model, X_test, y_test)

if __name__ == "__main__":
    run()
