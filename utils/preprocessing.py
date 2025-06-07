# utils/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def fill_missing_with_mean(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    数値カラムの欠損値を平均値で埋める

    Args:
        df (pd.DataFrame): 入力データフレーム
        columns (list): 平均埋めするカラム名リスト

    Returns:
        pd.DataFrame: 欠損値を埋めたデータフレーム
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def fill_missing_with_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    数値カラムの欠損値を中央値で埋める

    Args:
        df (pd.DataFrame): 入力データフレーム
        columns (list): 中央値埋めするカラム名リスト

    Returns:
        pd.DataFrame: 欠損値を埋めたデータフレーム
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def fill_missing_with_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    カテゴリカラムの欠損値を最頻値で埋める

    Args:
        df (pd.DataFrame): 入力データフレーム
        columns (list): 最頻値埋めするカラム名リスト

    Returns:
        pd.DataFrame: 欠損値を埋めたデータフレーム
    """
    df = df.copy()
    for col in columns:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
    return df


def universal_impute(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    SimpleImputerを使って数値カラムの欠損値を一括補完

    Args:
        df (pd.DataFrame): 入力データフレーム
        strategy (str): 'mean','median','most_frequent','constant'

    Returns:
        pd.DataFrame: 欠損値を補完したデータフレーム
    """
    df = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df


def drop_missing_rows(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:
    """
    欠損値が閾値以上ある行を削除

    Args:
        df (pd.DataFrame): 入力データフレーム
        thresh (float): 残す非欠損カラム比率

    Returns:
        pd.DataFrame: 行削除後のデータフレーム
    """
    min_count = int(thresh * df.shape[1])
    return df.dropna(thresh=min_count)


def label_encode(df: pd.DataFrame, columns: list) -> (pd.DataFrame, dict):
    """
    指定したカテゴリカラムをLabelEncoding

    Args:
        df (pd.DataFrame): 入力データフレーム
        columns (list): エンコード対象のカラム名リスト

    Returns:
        df (pd.DataFrame): エンコード後のデータフレーム
        encoders (dict): {カラム名: LabelEncoderオブジェクト}
    """
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    事前に学習したLabelEncoderを使ってカテゴリカラムを変換

    Args:
        df (pd.DataFrame): 入力データフレーム
        encoders (dict): {カラム名: LabelEncoderオブジェクト}

    Returns:
        pd.DataFrame: エンコード後のデータフレーム
    """
    df = df.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))
    return df


def one_hot_encode(df: pd.DataFrame, columns: list, drop_first: bool = False) -> pd.DataFrame:
    """
    指定したカテゴリカラムをOne-Hotエンコーディング

    Args:
        df (pd.DataFrame): 入力データフレーム
        columns (list): エンコード対象のカラム名リスト
        drop_first (bool): ダミー変数落としの有無

    Returns:
        pd.DataFrame: エンコード後のデータフレーム
    """
    df = df.copy()
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)
