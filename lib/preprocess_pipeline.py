import os
import pandas as pd
import numpy as np
import joblib

# utils modules
from utils.preprocessing import (
    fill_missing_ffill_bfill,
    fill_missing_with_median,
    fill_missing_with_mode,
    universal_impute,
    drop_missing_rows,
    standard_scale
)
from utils.timeseries import (
    add_time_features,
    add_lag_features,
    add_rolling_features
)
from utils.feature_selection import top_correlated_features
from utils.split import time_series_split
from utils.metrics import mae, rmse, mape


def preprocess(
    df: pd.DataFrame,
    target: str = 'price_actual',
    num_top_features: int = 50
) -> pd.DataFrame:
    """
    全体の前処理パイプラインを実行し、
    相関上位の特徴量に絞る。

    Args:
        df (pd.DataFrame): 生データ
        target (str): 目的変数名
        num_top_features (int): 相関上位何変数を使用するか

    Returns:
        pd.DataFrame: 前処理・特徴選択後のデータ
    """
    # 1. 欠損値補完
    df = fill_missing_ffill_bfill(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df = fill_missing_with_median(df, num_cols)
    df = fill_missing_with_mode(df, cat_cols)
    df = universal_impute(df, strategy='mean')
    df = drop_missing_rows(df, thresh=0.5)

    # 2. 時系列特徴量
    df = add_time_features(df, date_col='time')
    all_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df = add_lag_features(df, cols=[target] + num_cols, lags=[1, 24])
    df = add_rolling_features(df, cols=[target] + num_cols, windows=[3, 6, 24], stats=['mean', 'std'])

    # 3. 相関上位特徴量の選択
    top_feats = [f for f, _ in top_correlated_features(df, target, num_top_features)]
    # 必要に応じて時系列特徴量を保持
    mandatory_feats = [ 'hour', 'weekday', 'month', 'hour_sin', 'hour_cos' ]
    selected = list(set(top_feats + mandatory_feats + [target]))
    df = df[selected].dropna()

    return df
