
"""
Cluster features into groups using correlation + hierarchical clustering.
This attempts to recover "sensor groups" (temperature/vibration/pressure-like groups)
from anonymized features.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances


def cluster_features_by_correlation(X: pd.DataFrame, n_groups: int = 8, method='average'):
    """
    X: DataFrame of only features (no target), standardized.
    Returns: dict group_id -> list of columns
    """
    corr = X.corr().fillna(0)
    # Use distance = 1 - |corr|
    dist = 1 - np.abs(corr)
    # Convert to condensed distance matrix for linkage (must be symmetric with zeros on diag)
    condensed = squareform(dist.values)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=n_groups, criterion='maxclust')


    groups = {}
    for col, lab in zip(X.columns, labels):
        groups.setdefault(lab, []).append(col)
    return groups


def top_features_per_group(X: pd.DataFrame, groups: dict, top_k=10):
    # pick top variance features in each group
    out = {}
    for gid, cols in groups.items():
        variances = X[cols].var().sort_values(ascending=False)
        out[gid] = variances.index[:top_k].tolist()
    return out