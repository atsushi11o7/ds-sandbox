import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def top_correlated_features(
    df: pd.DataFrame,
    target_col: str,
    n_features: int = 10
) -> List[Tuple[str, float]]:
    """
    目的変数と相関の高い上位 n_features 個の説明変数を返す関数

    Args:
        df (pd.DataFrame): 説明変数と目的変数を含むデータフレーム
        target_col (str): 目的変数のカラム名
        n_features (int): 上位何個の特徴量を取得するか

    Returns:
        List[Tuple[str, float]]: (特徴量名, 相関係数) のタプルリスト、降順
    """
    # 数値カラムのみ抽出
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        raise ValueError(f"目的変数 {target_col} は数値型カラムとして存在しない")

    # 目的変数との相関係数を計算
    corrs = numeric_df.corr()[target_col].abs().sort_values(ascending=False)

    # 目的変数）以外の上位 n_features 個を取得
    top = corrs.drop(labels=[target_col]).head(n_features)

    return list(zip(top.index.tolist(), top.values.tolist()))


def plot_top_correlations(
    df: pd.DataFrame,
    target_col: str,
    n_features: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "coolwarm"
):
    """
    目的変数と相関の高い上位 n_features 個の説明変数をヒートマップで可視化

    Args:
        df (pd.DataFrame): 説明変数と目的変数を含むデータフレーム
        target_col (str): 目的変数のカラム名
        n_features (int): 上位何個の特徴量を可視化するか
        figsize (Tuple[int, int]): プロットサイズ
        cmap (str): ヒートマップのカラーマップ
    """
    # 相関上位を取得
    top_features = [f for f, _ in top_correlated_features(df, target_col, n_features)]
    subset = df[top_features + [target_col]].select_dtypes(include=[np.number])

    # 相関行列を計算
    corr_matrix = subset.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5
    )
    plt.title(f"Top {n_features} features correlated with '{target_col}'")
    plt.show()