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
