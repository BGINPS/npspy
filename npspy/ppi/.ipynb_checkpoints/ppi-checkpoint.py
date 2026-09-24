#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: ppi.py
@Description: description of this file
@Datatime: 2025/06/23 10:15:58
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd

from . import tools as tl

def define_state_for_a_read_obj(
    read_obj: dict,
    low_signal_min_len: int = 50,
):
    high_mean, low_mean = get_high_mean_and_low_mean_for_a_read_obj(
        read_obj=read_obj,
        low_signal_min_len=low_signal_min_len,
    )
    
    if high_mean == 0 and low_mean != 0:
        return 'strong'
    if high_mean != 0 and low_mean != 0:
        return 'weak'
    if high_mean != 0 and low_mean == 0:
        return 'non_bind'
    return 'unknown'


def get_high_mean_and_low_mean_for_a_read_obj(
    read_obj: dict,
    low_signal_min_len: int = 50,
):
    signal = tl.norm_by_mean_polyT_i2io_for_a_read_obj(
        read_obj=read_obj
    )

    signal = signal[10:-10]

    high_signal = signal[signal > 0.4]
    low_signal = signal[signal < 0.2]
    
    if len(high_signal) > 0:
        high_mean = np.mean(high_signal - 0.4)
    else:
        high_mean = 0.0
    
    if len(low_signal) >= low_signal_min_len:
        low_mean = np.mean(0.2 - low_signal)
    else:
        low_mean = 0.0
    
    return high_mean, low_mean


def split_an_obj_into_three_objs(
    obj: dict,
    low_signal_min_len: int = 1000,
):
    strong_obj, weak_obj, non_bind_obj = {}, {}, {}
    for read_id, read_obj in obj.items():
        state = define_state_for_a_read_obj(
            read_obj=read_obj,
            low_signal_min_len=low_signal_min_len,
        )
        if state == 'strong':
            strong_obj[read_id] = read_obj
        elif state == 'weak':
            weak_obj[read_id] = read_obj
        elif state == 'non_bind':
            non_bind_obj[read_id] = read_obj
    return strong_obj, weak_obj, non_bind_obj


def get_high_mean_and_low_mean_for_an_obj(
    obj: dict,
    low_signal_min_len: int = 1000,
):
    stat_df = []
    for read_id, read_obj in obj.items():
        high_mean, low_mean = get_high_mean_and_low_mean_for_a_read_obj(
            read_obj=read_obj,
            low_signal_min_len=low_signal_min_len,
        )
        stat_df.append([read_id, high_mean, low_mean])
    stat_df = pd.DataFrame(stat_df, columns=['read_id', 'high_mean', 'low_mean'])
    stat_df.set_index('read_id', inplace=True)
    return stat_df



def get_length_of_low_points_for_a_read_obj(
    read_obj: dict,
):
    signal = tl.norm_by_mean_polyT_i2io_for_a_read_obj(
        read_obj=read_obj
    )

    signal = signal[10:-10]

    # high_signal = signal[signal > 0.4]
    low_signal = signal[signal < 0.2]

    return len(low_signal)

def get_length_of_low_points_for_an_obj(
    obj: dict,
):
    lengths = []
    for read_id, read_obj in obj.items():
        length = get_length_of_low_points_for_a_read_obj(
            read_obj=read_obj
        )
        lengths.append([read_id, length])
    lengths = pd.DataFrame(lengths, columns=['read_id', 'length'])
    lengths.set_index('read_id', inplace=True)
    return lengths