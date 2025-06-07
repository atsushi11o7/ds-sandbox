# utils/timeseries.py

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, date_col: str = 'time') -> pd.DataFrame:
    """
    日付情報から時系列特徴量を追加する関数
    以下のカラムを追加
      - hour, weekday, month
      - hour_sin, hour_cos (サイクリック変換)

    Args:
        df (pd.DataFrame): 元のデータフレーム、date_colの存在が必須
        date_col (str): 日付カラム名

    Returns:
        pd.DataFrame: 特徴量を追加したデータフレーム
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['hour'] = df[date_col].dt.hour
    df['weekday'] = df[date_col].dt.weekday
    df['month'] = df[date_col].dt.month
    # サイクリック変換
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: list,
    lags: list = [1, 24]
) -> pd.DataFrame:
    """
    指定した数値カラムに対し遅延(ラグ)特徴量を追加

    Args:
        df (pd.DataFrame): 元のデータフレーム（時系列順にソートされていることが望ましい）
        cols (list): ラグを追加するカラム名のリスト
        lags (list): 何時点前のラグを作成するかのリスト

    Returns:
        pd.DataFrame: 遅延特徴量を追加したデータフレーム
    """
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: list,
    windows: list = [3, 6, 24],
    stats: list = ['mean', 'std']
) -> pd.DataFrame:
    """
    指定した数値カラムに対し移動統計量特徴量を追加

    Args:
        df (pd.DataFrame): 元のデータフレーム（時系列順）
        cols (list): 移動統計を計算するカラム名
        windows (list): 窓幅（時間幅）のリスト
        stats (list): 計算する統計量 ("mean", "std", "min", "max").

    Returns:
        pd.DataFrame: 移動統計量を追加したデータフレーム
    """
    df = df.copy()
    for col in cols:
        for w in windows:
            roll = df[col].rolling(window=w)
            if 'mean' in stats:
                df[f'{col}_roll_mean_{w}'] = roll.mean()
            if 'std' in stats:
                df[f'{col}_roll_std_{w}'] = roll.std()
            if 'min' in stats:
                df[f'{col}_roll_min_{w}'] = roll.min()
            if 'max' in stats:
                df[f'{col}_roll_max_{w}'] = roll.max()
    return df