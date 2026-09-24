#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: tools.py
@Description: description of this file
@Datatime: 2025/06/16 14:50:45
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, jaccard_score, accuracy_score
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"

def get_diff_for_a_read_obj(
    read_obj: dict,
    only_window: bool = True,
) -> np.ndarray:
    """
    Calculate the difference of the signal for a given read object.
    Args:
        read_obj (dict): A dictionary containing the read object with keys 'signal' and 'window'.
    Returns:
        np.ndarray: The difference of the signal within the specified window.
    """
    if 'signal' not in read_obj or 'window' not in read_obj:
        raise ValueError("The read object must contain 'signal' and 'window' keys.")
    if not isinstance(read_obj['signal'], np.ndarray):
        raise TypeError("The 'signal' in the read object must be a numpy array.")
    if not isinstance(read_obj['window'], (list, tuple)) or len(read_obj['window']) != 2:
        raise ValueError("The 'window' in the read object must be a list or tuple of length 2.")

    signal = read_obj['signal']
    s, e = read_obj['window'][0], read_obj['window'][1]
    if only_window:
        signal = signal[s:e]
    signal_diff = np.diff(signal)
    return signal_diff.astype(np.float32)

def get_diff_mean_and_var_for_each_read_in_an_obj(
    obj: dict,
) -> pd.DataFrame:
    """
    Calculate the mean and variance of the difference of signals for each read in an object.
    Args:
        obj (dict): A dictionary where keys are read IDs and values are read objects containing 'signal' and 'window'.
    Returns:
        pd.DataFrame: A DataFrame with read IDs as index, and columns for mean and variance of the signal differences.
    """
    df = []
    for read_id, read_obj in obj.items():
        diff_signal = get_diff_for_a_read_obj(read_obj)
        df.append([read_id, np.mean(diff_signal), np.var(diff_signal)])
    df = pd.DataFrame(df, columns=['read_id', 'mean', 'var'])
    df.set_index('read_id', inplace=True)
    return df


def get_iqr_cutoff(
    a: np.array, # 1d
    iqr_time: float = 1.5,
) -> float:
    """
    Calculate the Interquartile Range (IQR) upper cutoff for a given array.
    Args:
        a (np.array): A 1D numpy array of numerical values.
        iqr_time (float): A multiplier for the IQR to determine the cutoff.
    Returns:
        float: The IQR cutoff value.
    """
    iqr = np.quantile(a, 0.75) - np.quantile(a, 0.25)
    return np.quantile(a, 0.75) + iqr * iqr_time


def get_mad_cutoff(
    a: np.array, # 1d
    mad_time: float = 2.0,
) -> float:
    """ Calculate the Median Absolute Deviation (MAD) upper cutoff for a given array.
    Args:
        a (np.array): A 1D numpy array of numerical values. 
        mad_time (float): A multiplier for the MAD to determine the cutoff.
    Returns:
        float: The MAD cutoff value.
    """
    median_ = np.median(a)
    mad = np.median(np.abs(a - median_))
    return median_ + mad_time * mad


def safe_random_sample(arr, n_samples=200):
    """
    从数组中随机抽取n_samples行，不足则取全部
    参数：
        arr: 输入NumPy数组
        n_samples: 目标采样数（默认200）
    返回：
        采样后的子数组
    """
    if len(arr) <= n_samples:
        return arr.copy()
    return arr[np.random.choice(arr.shape[0], n_samples, replace=False)]


def get_peak_x_for_by_density_plot(
    a: np.ndarray, # 1d
):
    ax = sns.kdeplot(a)
    line = ax.get_lines()[0]  # 获取第一条线（密度线）
    x_data = line.get_xdata()  # x坐标数据（数据点的值）
    y_data = line.get_ydata()   # y坐标数据（密度值）

    # 找到密度最大值对应的索引
    peak_idx = np.argmax(y_data)
    peak_x = x_data[peak_idx]
    peak_y = y_data[peak_idx]
    plt.close()
    return peak_x



def binarize(series):
    threshold = np.median(series)
    return np.where(series > threshold, 1, 0)

def lempel_ziv_complexity(binary_sequence):
    vocabulary = {tuple()}
    i, complexity = 0, 0
    
    while i < len(binary_sequence):
        j = i + 1
        while j <= len(binary_sequence):
            # 检查当前子序列是否在词汇表中
            subsequence = tuple(binary_sequence[i:j])
            if subsequence not in vocabulary:
                vocabulary.add(subsequence)
                complexity += 1
                break
            j += 1
        i = j - 1
    return complexity

def normalized_lzc(sequence):
    binary_seq = binarize(sequence)
    lz = lempel_ziv_complexity(binary_seq)
    return lz / (len(sequence) / np.log(len(sequence)))

def get_lzc_without_norm_by_len(
    a: np.ndarray, # 1d
):
    """ Calculate the Lempel-Ziv complexity (LZC) of a 1D numpy array without normalization by length.
    Args:
        a (np.ndarray): A 1D numpy array of numerical values.
    Returns:
        float: The Lempel-Ziv complexity of the binary representation of the array.
    """
    binary_seq = binarize(a)
    lz = lempel_ziv_complexity(binary_seq)
    return lz

def get_lzc_for_each_line_in_a_dataframe(
    data_df: pd.DataFrame,
    batch_size: int = 1000,
    n_jobs: int = 10,
) -> pd.DataFrame:
    arr = data_df.to_numpy()
    batches = [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]

    with Pool(n_jobs) as pool:
        results = pool.map(get_lzc_without_norm_by_len_batch, batches)
    
    res = np.concatenate(results)

    lzc_df = pd.DataFrame({'lzc': res}, index=data_df.index)
    return lzc_df

def get_lzc_without_norm_by_len_batch(
    one_batch: np.ndarray, # 2d
):
    return [get_lzc_without_norm_by_len(row) for row in one_batch]



def get_acc_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
):
    true_ = pred_proba_df.iloc[:,-1].values
    pred_ = pred_proba_df.iloc[:,:-1].values
    pred_ = np.argmax(pred_, axis=1)
    # acc = np.mean(pred_ == true_)
    acc = accuracy_score(true_, pred_)
    return acc

def get_precision_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
):
    true_ = pred_proba_df.iloc[:,-1].values
    pred_ = pred_proba_df.iloc[:,:-1].values
    pred_ = np.argmax(pred_, axis=1)
    precision = precision_score(true_, pred_, average='macro')
    return precision

def get_recall_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
):
    true_ = pred_proba_df.iloc[:,-1].values
    pred_ = pred_proba_df.iloc[:,:-1].values
    pred_ = np.argmax(pred_, axis=1)
    recall = recall_score(true_, pred_, average='macro')
    return recall

def get_f1_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
):
    true_ = pred_proba_df.iloc[:,-1].values
    pred_ = pred_proba_df.iloc[:,:-1].values
    pred_ = np.argmax(pred_, axis=1)
    f1 = f1_score(true_, pred_, average='macro')
    return f1

#AUROC == ROC-AUC
def get_auroc_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
):
    auroc = roc_auc_score(pred_proba_df.iloc[:,-1].values, pred_proba_df.iloc[:,:-1].values, multi_class='ovr')
    return auroc


def get_jaccard_score_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
):
    true_ = pred_proba_df.iloc[:,-1].values
    pred_ = pred_proba_df.iloc[:,:-1].values
    pred_ = np.argmax(pred_, axis=1)
    js = jaccard_score(true_, pred_, average='macro')
    return js


def get_metrics_from_pred_proba_df(
    pred_proba_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate various metrics from a DataFrame of predicted probabilities.
    Args:
        pred_proba_df (pd.DataFrame): DataFrame containing predicted probabilities and true labels.
    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics.
    """
    metrics = {
        'accuracy': [get_acc_from_pred_proba_df(pred_proba_df)],
        'precision': [get_precision_from_pred_proba_df(pred_proba_df)],
        'recall': [get_recall_from_pred_proba_df(pred_proba_df)],
        'f1_score': [get_f1_from_pred_proba_df(pred_proba_df)],
        'jaccard_score': [get_jaccard_score_from_pred_proba_df(pred_proba_df)],
        #'auroc': [get_auroc_from_pred_proba_df(pred_proba_df)],
    }

    metrics_df = pd.DataFrame(metrics)

    return metrics_df


def get_metrics_from_pred_df(
    pred_df: pd.DataFrame,
    true_col: str = 'true',
    pred_col: str = 'pred',
) -> pd.DataFrame:
    true_ = pred_df[true_col].values
    pred_ = pred_df[pred_col].values
    metrics = {
        'accuracy': [accuracy_score(true_, pred_)],
        'precision': [precision_score(true_, pred_, average='macro')],
        'recall': [recall_score(true_, pred_, average='macro')],
        'f1_score': [f1_score(true_, pred_, average='macro')],
        'jaccard_score': [jaccard_score(true_, pred_, average='macro')],
    }
    metrics_df = pd.DataFrame(metrics)
    return metrics_df