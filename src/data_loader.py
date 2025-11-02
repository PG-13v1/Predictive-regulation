import pandas as pd
from pathlib import Path
from .config import DATA_DIR


def load_secom(path: str = None):
    p = Path(path) if path else DATA_DIR / 'secom_combined.csv'
    df = pd.read_csv(p)
    
    # standardize target column names if needed
    # Possible target column names: 'Pass/Fail', 'class', 'target'
    for c in ['Pass/Fail', 'Pass_Fail', 'target', 'class']:
        if c in df.columns:
            df = df.rename(columns={c: 'target'})
            break
    
    if 'target' not in df.columns:
        raise ValueError('Target column not found; expected "Pass/Fail" or similar')
    
    # map target values to 0/1
    vals = df['target'].unique()
    
    # If labels are -1 / 1
    if set(vals) == {-1, 1}:
        df['target'] = df['target'].map({-1: 0, 1: 1})
    else:
        df['target'] = df['target'].astype(int)
    
    return df