#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: tools.py
@Description: description of this file
@Datatime: 2025/06/23 09:19:11
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd

# from .ppi import norm_by_mean_polyT_i2io_for_a_read_obj

def define_if_has_polyT_for_a_read_obj(
    read_obj: dict,
    window_start_percent: float = 0.1,
    window_end_percent: float = 0.2,
    min_mean_of_polyT_i2io: float = 0.26,
    max_mean_of_polyT_i2io: float = 0.32,
    max_std_of_polyT_i: float = 4.0
) -> bool:
    res = get_mean_of_polyT_i2io_for_a_read_obj(
        read_obj=read_obj,
        window_start_percent=window_start_percent,
        window_end_percent=window_end_percent,
        min_mean_of_polyT_i2io=min_mean_of_polyT_i2io,
        max_mean_of_polyT_i2io=max_mean_of_polyT_i2io,
        max_std_of_polyT_i=max_std_of_polyT_i
    )

    if res is None:
        return False
    else:
        return True


def get_mean_of_polyT_i2io_for_a_read_obj(
    read_obj: dict,
    window_start_percent: float = 0.1,
    window_end_percent: float = 0.2,
    min_mean_of_polyT_i2io: float = 0.26,
    max_mean_of_polyT_i2io: float = 0.32,
    max_std_of_polyT_i: float = 4.0
) -> Union[float, None]:
    """
    Calculate the mean of polyT i2io for a given read object.
    Args:
        read_obj (dict): A dictionary containing the read data, including 'window', 'signal', and 'OpenPore'.
        window_start_percent (float): The starting percentage of the window to consider for polyT signal.
        window_end_percent (float): The ending percentage of the window to consider for polyT signal.
        min_mean_of_polyT_i2io (float): Minimum mean of polyT i2io to consider valid.
        max_mean_of_polyT_i2io (float): Maximum mean of polyT i2io to consider valid.
        max_std_of_polyT_i (float): Maximum standard deviation of polyT signal to consider valid.
    Returns:
        float or None: The mean of polyT i2io if valid, otherwise None.
    """
    s, e = read_obj['window']
    signal = read_obj['signal'][s:e]
    polyT_signal = signal[int(len(signal) * window_start_percent):int(len(signal) * window_end_percent)]
    polyT_signal = polyT_signal.astype(np.float64)  # Ensure the signal is in float64 format for division
    polyT_i2io = polyT_signal / read_obj['OpenPore']
    mean_polyT_i2io = np.mean(polyT_i2io)
    std_polyT_i = np.std(polyT_signal)
    if (mean_polyT_i2io < min_mean_of_polyT_i2io or
        mean_polyT_i2io > max_mean_of_polyT_i2io or
        std_polyT_i > max_std_of_polyT_i):
        return None
    return mean_polyT_i2io


def get_polyT_i2io_for_an_obj(
    obj: dict,
    window_start_percent: float = 0.1,
    window_end_percent: float = 0.2,
    min_mean_of_polyT_i2io: float = 0.26,
    max_mean_of_polyT_i2io: float = 0.32,
    max_std_of_polyT_i: float = 4.0
):
    """
    Calculate the mean of polyT i2io for each read in the given object.
    Args:
        obj (dict): A dictionary where keys are read IDs and values are read objects containing 'window', 'signal', and 'OpenPore'.
        window_start_percent (float): The starting percentage of the window to consider for polyT signal.
        window_end_percent (float): The ending percentage of the window to consider for polyT signal.
        min_mean_of_polyT_i2io (float): Minimum mean of polyT i2io to consider valid.
        max_mean_of_polyT_i2io (float): Maximum mean of polyT i2io to consider valid.
        max_std_of_polyT_i (float): Maximum standard deviation of polyT signal to consider valid.
    Returns:
        dict: A new dictionary with read IDs as keys and read objects with 'polyT_i2io' added if valid.
    """
    new_obj = {}
    for read_id, read_obj in obj.items():
        mean_polyT_i2io = get_mean_of_polyT_i2io_for_a_read_obj(
            read_obj=read_obj,
            window_start_percent=window_start_percent,
            window_end_percent=window_end_percent,
            min_mean_of_polyT_i2io=min_mean_of_polyT_i2io,
            max_mean_of_polyT_i2io=max_mean_of_polyT_i2io,
            max_std_of_polyT_i=max_std_of_polyT_i
        )
        if mean_polyT_i2io is not None:
            read_obj['polyT_i2io'] = mean_polyT_i2io
            new_obj[read_id] = read_obj
    return new_obj


def norm_by_mean_polyT_i2io_for_a_read_obj(
    read_obj: dict,
    polyT_target_signal: float = 0.3,
) -> np.ndarray:
    """
    
    """
    s, e = read_obj['window']
    signal = read_obj['signal'].astype(np.float32) / read_obj['OpenPore']
    signal = signal[s:e]
    polyT_i2io = read_obj['polyT_i2io']
    signal = signal / polyT_i2io * polyT_target_signal
    return signal



def get_features_low_salt(
    signal: np.ndarray,
    name: str = None,
) -> pd.DataFrame:

    feature_names = []
    features = []
    
    for i in np.arange(0.2, 0.401, 0.01):
        feature_names.append(f'low_than_{i:.2f}')
        features.append(np.mean(signal<=i))

    for i in np.arange(0.4, 0.601, 0.01):
        feature_names.append(f'high_than_{i:.2f}')
        features.append(np.mean(signal>=i))

    feature_names.append('low_0.3_to_high_0.5')
    features.append(max(np.sum(signal<=0.3), 1) / max(np.sum(signal>=0.5), 1))

    feature_names.append('low_0.2_to_high_0.6')
    features.append(max(np.sum(signal<=0.2), 1) / max(np.sum(signal>=0.6), 1))

    feature_names.append('mean_of_low_0.4')
    features.append(np.mean(signal[signal<0.4]))

    feature_names.append('mean_of_high_0.4')
    features.append(np.mean(signal[signal>0.4]))
    
    df = pd.DataFrame({'features': features}, index=feature_names).T

    if name:
        df.index = [name]
    
    return df


def get_features_high_salt(
    signal: np.ndarray,
    name: str = None,
) -> pd.DataFrame:

    feature_names = []
    features = []
    
    for i in np.arange(0.1, 0.301, 0.01):
        feature_names.append(f'low_than_{i:.2f}')
        features.append(np.mean(signal<=i))

    for i in np.arange(0.3, 0.501, 0.01):
        feature_names.append(f'high_than_{i:.2f}')
        features.append(np.mean(signal>=i))

    feature_names.append('low_0.22_to_high_0.40')
    features.append(max(np.sum(signal<=0.22), 1) / max(np.sum(signal>=0.4), 1))

    # feature_names.append('low_0.1_to_high_0.5')
    # features.append(max(np.sum(signal<=0.1), 1) / max(np.sum(signal>=0.5), 1))

    # feature_names.append('mean_of_low_0.3')
    # features.append(np.mean(signal[signal<0.3]))

    # feature_names.append('mean_of_high_0.3')
    # features.append(np.mean(signal[signal>0.3]))
    
    df = pd.DataFrame({'features': features}, index=feature_names).T

    if name:
        df.index = [name]
    
    return df

def trim_array_percentiles(arr, lower_pct=5, upper_pct=95):
    """
    截取数组指定百分位区间(默认去除前5%和后5%)
    参数:
        arr: 输入数组
        lower_pct: 下限百分比(默认5)
        upper_pct: 上限百分比(默认95)
    返回:
        截取后的子数组
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    lower_idx = int(len(arr) * lower_pct / 100)
    upper_idx = int(len(arr) * upper_pct / 100)
    
    return arr[lower_idx:upper_idx]


def get_features_for_an_obj(
    obj: dict,
    polyT_target_signal: float,
    lib_type: Literal['high_salt', 'low_salt'],
):
    df = []
    for read_id, read_obj in obj.items():
        signal = norm_by_mean_polyT_i2io_for_a_read_obj(read_obj, polyT_target_signal=polyT_target_signal)
        signal = trim_array_percentiles(signal)
        if lib_type == 'high_salt':
            df.append(get_features_high_salt(signal, read_id))
        elif lib_type == 'low_salt':
            df.append(get_features_low_salt(signal, read_id))
    df = pd.concat(df)

    return df


def get_low_high_ratio_for_an_obj(
    obj: dict,
    polyT_target_signal: float,
    low_cutoff: float,
    high_cutoff: float,
):
    df = []
    for read_id, read_obj in obj.items():
        signal = norm_by_mean_polyT_i2io_for_a_read_obj(read_obj, polyT_target_signal=polyT_target_signal)
        signal = trim_array_percentiles(signal)
        ratio = max(np.sum(signal<=low_cutoff), 1) / max(np.sum(signal>=high_cutoff), 1)
        df.append(pd.DataFrame({'low_high_ratio': [ratio]}, index=[read_id]))
    df = pd.concat(df)

    df['log1p_low_high_ratio'] = np.log1p(df['low_high_ratio'])

    return df