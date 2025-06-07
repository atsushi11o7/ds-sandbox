# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_time_series(
    df: pd.DataFrame,
    cols: list,
    date_col: str = 'time',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    時系列プロットを描画

    Args:
        df (pd.DataFrame): データフレーム
        cols (list): プロットするカラム名リスト
        date_col (str): 日付インデックスまたは日付カラム
        figsize (tuple): 図のサイズ
    """
    df_plot = df.copy()
    if date_col in df_plot.columns:
        df_plot = df_plot.set_index(pd.to_datetime(df_plot[date_col]))
    df_plot[cols].plot(figsize=figsize)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Plot')
    plt.show()


def plot_missing_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    欠損値の有無をヒートマップで可視化

    Args:
        df (pd.DataFrame): データフレーム
        figsize (tuple): 図のサイズ
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Data Heatmap')
    plt.show()


def plot_distribution(
    df: pd.DataFrame,
    cols: list,
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    指定カラムの分布（ヒストグラム）をプロット

    Args:
        df (pd.DataFrame): データフレーム
        cols (list): 分布を描画するカラム名
        bins (int): ビン数
        figsize (tuple): 図のサイズ
    """
    df[cols].hist(bins=bins, figsize=figsize)
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'vlag'
) -> None:
    """
    指定カラム間の相関行列をヒートマップで可視化

    Args:
        df (pd.DataFrame): データフレーム
        cols (list): 相関を計算するカラム名
        figsize (tuple): 図のサイズ
        cmap (str): カラーマップ
    """
    corr = df[cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap,
                cbar_kws={'shrink': 0.8}, linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
