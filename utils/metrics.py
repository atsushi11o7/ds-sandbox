# utils/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均絶対誤差 (MAE) を計算

    Args:
        y_true (np.ndarray): 実測値
        y_pred (np.ndarray): 予測値

    Returns:
        float: MAE
    """
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    二乗平均平方根誤差 (RMSE) を計算

    Args:
        y_true (np.ndarray): 実測値
        y_pred (np.ndarray): 予測値

    Returns:
        float: RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均絶対パーセント誤差 (MAPE) を計算

    Args:
        y_true (np.ndarray): 実測値
        y_pred (np.ndarray): 予測値

    Returns:
        float: MAPE (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
