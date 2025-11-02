import tensorflow as tf
from tensorflow.keras import layers, models
from .config import LSTM_LATENT


def build_lstm_autoencoder(n_features, latent_dim=LSTM_LATENT):
    """
    Build an LSTM autoencoder for sequence data.
    
    Args:
        n_features (int): Number of features in the input data.
        latent_dim (int, optional): Dimensionality of the latent space. Defaults to LSTM_LATENT.
    
    Returns:
        models.Model: Trained LSTM autoencoder model.
    """
    input_layer = layers.Input(shape=(n_features, 1), name='input')
    
    # Encoder
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    # Decoder
    x = layers.RepeatVector((n_features // 4))(latent)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    
    # final reconstruction to single channel
    decoded = layers.Conv1D(1, kernel_size=3, padding='True', activation='sigmoid', name='output')(x)