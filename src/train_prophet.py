import os
from joblib import dump
from .data_loader import load_secom
from .preprocess import basic_impute_and_scale
from .feature_groups import cluster_features_by_correlation
from .prophet_wrapper import train_prophet_for_group
from .utils import get_logger


logger = get_logger('train_prophet')

def run(out_dir: str = 'models/prophet_models', n_groups: int = 8):
  os.makedirs(out_dir, exist_ok=True)
  df = load_secom()
  Xs = basic_impute_and_scale(df)
  features = Xs.drop(columns=['target'])
  groups = cluster_features_by_correlation(features, n_groups=n_groups)
  dump(groups, os.path.join(out_dir, 'feature_groups.joblib'))


  for gid, cols in groups.items():
    X_group = features[cols]
    model = train_prophet_for_group(X_group, os.path.join(out_dir, f'prophet_group_{gid}.joblib'))
    logger.info(f'Trained prophet for group {gid} with {len(cols)} features')

