# utils/split.py

from sklearn.model_selection import TimeSeriesSplit


def time_series_split(df, n_splits: int = 5):
    """
    時系列データ用のクロスバリデーションイテレータを生成

    Args:
        df (pd.DataFrame): 対象データフレーム（インデックス順に時系列として扱う）
        n_splits (int): 分割数

    Yields:
        (train_index, test_index): 学習と検証のインデックスタプル
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(df):
        yield train_idx, test_idx


def split_and_scale(
    df: pd.DataFrame,
    target: str = 'price_actual',
    test_size: float = 0.1
):
    """
    データを学習用と検証用に分割し、標準化を行う。

    Returns:
        X_train, X_val, y_train, y_val, scaler
    """
    # 時系列分割
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_val = val.drop(columns=[target])
    y_val = val[target]

    # 標準化
    X_train_scaled, scaler = standard_scale(X_train)
    X_val_scaled, _ = standard_scale(X_val, scaler)

    return X_train_scaled, X_val_scaled, y_train.values, y_val.values, scaler