import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from .config import VAR_THRESH


def basic_impute_and_scale(df: pd.DataFrame, save_scaler: str = None) -> pd.DataFrame:
    X = df.drop(columns=['target'])
    colnames = X.columns.tolist()

    # 1) Remove constant columns
    vt = VarianceThreshold(threshold=VAR_THRESH)
    vt.fit(X.fillna(0))
    keep_mask = vt.get_support()
    X = X.iloc[:, keep_mask]
    kept_cols = X.columns.tolist()

    # 2) Impute
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # 3) Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    if save_scaler:
        save_dir = Path(save_scaler).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({'imputer': imputer, 'scaler': scaler, 'cols': kept_cols}, save_scaler)

    X_scaled_df = pd.DataFrame(X_scaled, columns=kept_cols)
    X_scaled_df['target'] = df['target'].values
    return X_scaled_df


def print_head(path: str, n: int = 5) -> None:
    try:
        print(f"\n--- First {n} lines of {path} ---")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i in range(n):
                line = f.readline()
                if not line:
                    break
                print(repr(line.rstrip("\n\r")))
    except FileNotFoundError as e:
        print(f"Could not print head of {path}: {e}")