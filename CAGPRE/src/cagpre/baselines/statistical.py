import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def evaluate_global_iqr(
    df_train: pd.DataFrame, df_test: pd.DataFrame, return_ranges: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
    from ..engines.shared import compute_metrics, compute_iou
    q1 = df_train["value"].quantile(0.25)
    q3 = df_train["value"].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    iqr_low = np.full(len(df_test), low)
    iqr_high = np.full(len(df_test), high)
    y_pred = (df_test["value"].values < low) | (df_test["value"].values > high)
    metrics = compute_metrics(df_test["is_error"].values, y_pred.astype(int))
    metrics["iou"] = compute_iou(iqr_low, iqr_high, df_test["gt_low"].values, df_test["gt_high"].values)
    if return_ranges:
        return metrics, iqr_low, iqr_high
    return metrics

def evaluate_isolation_forest(
    x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int
) -> Dict[str, float]:
    from ..engines.shared import compute_metrics
    model = IsolationForest(random_state=seed, contamination=0.05)
    model.fit(x_train)
    preds = model.predict(x_test)
    y_pred = (preds == -1).astype(int)
    return compute_metrics(y_test, y_pred)

def evaluate_lof(
    x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    from ..engines.shared import compute_metrics
    model = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=0.05)
    model.fit(x_train)
    preds = model.predict(x_test)
    y_pred = (preds == -1).astype(int)
    return compute_metrics(y_test, y_pred)

