"""
Train Prophet models per-feature-group by aggregating group mean time-series.
Note: SECOM is not inherently temporal; if you have timestamped runs use that column.
Here we demonstrate per-group drift modelling by treating sample index as time.
"""
import pandas as pd
from prophet import Prophet
from joblib import dump
from .utils import get_logger


logger = get_logger('prophet')


def make_group_timeseries(X_group: pd.DataFrame) -> pd.DataFrame:
    """
    Create a time series DataFrame from the given feature data.

    Args:
        X_group (pd.DataFrame): DataFrame of features (no target).

    Returns:
        pd.DataFrame: Time series DataFrame with a 'ds' column and a 'y' column.
    """
    # Create a 'time index' as an integer sequence
    ts = pd.DataFrame({
        'ds': pd.date_range(start='2000-01-01', periods=len(X_group), freq='D'),
        'y': X_group.mean(axis=1).values # aggregate by mean
    })
    return ts


def train_prophet_for_group(X_group: pd.DataFrame, out_path: str) -> Prophet:
    """
    Train a Prophet model for the given feature group.

    Args:
        X_group (pd.DataFrame): DataFrame of features (no target).
        out_path (str): Path to save the trained Prophet model.

    Returns:
        Prophet: Trained Prophet model.
    """
    ts = make_group_timeseries(X_group)
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(ts)
    dump(m, out_path)
    logger.info(f'Saved Prophet model to {out_path}')
    return m