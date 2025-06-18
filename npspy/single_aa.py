#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: single_aa.py
@Description: description of this file
@Datatime: 2024/12/17 10:20:54
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

import pickle
import numpy as np
import pandas as pd
import copy
import os
import re
import glob
from typing import Dict, Optional, Tuple, Union, Literal, List
from scipy.signal import find_peaks

from . import io

def _get_reads_from_dat_derived_signals(
    signals: np.array,
    i0_min_c: float = 200,
    i0_max_c: float = 250,
    i0_min_len: int = 200,
    cut_signal_tail: int = 50,
    target_signal_min_len: int = 1000,
    read_id_prefix: str = 'read'
):
    peaks = _find_peaks(x=signals, height=500, distance=50000, inverse=True)
    peak_segments = np.array([[i, i+1] for i in peaks], dtype=np.int32)
    i0_segments = _find_i0_platforms(x=signals, min_c=i0_min_c,
                                     max_c=i0_max_c,
                                     min_len=i0_min_len,
                                     )
    if len(i0_segments) == 0:
        return {}
    peak_seg_df = pd.DataFrame(peak_segments, columns=['start', 'end'])
    peak_seg_df['label'] = 'peak_seg'
    i0_seg_df = pd.DataFrame(i0_segments, columns=['start', 'end'])
    i0_seg_df['label'] = 'I0_seg'
    df = pd.concat([peak_seg_df, i0_seg_df], ignore_index=True).sort_values(by='start').reset_index(drop=True)

    all_reads, all_read_coors, types = _get_reads_from_sem_df(df, signals=signals, i0_len=i0_min_len)

    obj = {}
    for one_read, one_coor, one_type in zip(all_reads, all_read_coors, types):
        one_read = one_read[0:-cut_signal_tail]
        if len(one_read) >= target_signal_min_len:
            read_id = f'{read_id_prefix}_{one_coor}_{one_type}'
            obj[read_id] = {}
            obj[read_id]['signal'] = one_read
            obj[read_id]['OpenPore'] = np.median(one_read[0:i0_min_len])
            obj[read_id]['type'] = one_type
            obj[read_id]['window'] = [i0_min_len,len(one_read)-1]

    return obj
    

def get_obj_from_dat_files(
    dir_path_include_all_channels: str,
    i0_min_c: float = 200,
    i0_max_c: float = 250,
    i0_min_len: int = 200,
    cut_signal_tail: int = 50,
    target_signal_min_len: int = 1000,
):
    all_channel_dirs = glob.glob(f'{dir_path_include_all_channels}/channel*')
    obj = {}
    for one_channel_dir in all_channel_dirs:
        signals = io.read_all_dat_files_in_a_channel(channel_path=one_channel_dir, sort_dat_files=True)
        tmp = re.search(r'\/(20\d+)_LAB.*(channel\d+)', one_channel_dir)
        run_id, channel_id = tmp.group(1), tmp.group(2)
        read_id_prefix = f'{run_id}_{channel_id}'
        one_obj = _get_reads_from_dat_derived_signals(
            signals=signals,
            i0_min_c=i0_min_c,
            i0_max_c=i0_max_c,
            i0_min_len=i0_min_len,
            cut_signal_tail=cut_signal_tail,
            target_signal_min_len=target_signal_min_len,
            read_id_prefix=read_id_prefix
        )
        if len(one_obj) == 0:
            print(f'!!!{read_id_prefix} has no read, please check!!!')
        obj.update(one_obj)
    
    return obj        
        

    

def _get_reads_from_sem_df(
    df: pd.DataFrame,
    signals: np.array,
    i0_len: int,
):
    all_reads, all_read_coors, types = [], [], []
    for i in range(1, len(df)):
        pre_row = df.iloc[i-1,:]
        now_row = df.iloc[i,:]
        if now_row['label'] == 'I0_seg' and pre_row['label'] == 'I0_seg':
            start = pre_row['end'] - i0_len
            end = now_row['start']
            types.append('type2')
        
        if now_row['label'] == 'I0_seg' and pre_row['label'] == 'peak_seg':
            continue

        if now_row['label'] == 'peak_seg' and pre_row['label'] == 'I0_seg':
            start = pre_row['end'] - i0_len
            end = now_row['start']
            types.append('type1')

        if now_row['label'] == 'peak_seg' and pre_row['label'] == 'peak_seg':
            continue

        all_reads.append(signals[start:end])
        all_read_coors.append(f'start_end_{start}_{end}')

    return all_reads, all_read_coors, types



        
    


def _find_peaks(
    x: np.array,
    height: float,
    distance: float,
    inverse: bool = False,
):
    if inverse:
        x = x * (-1)
    peaks, _ = find_peaks(x, height=height, distance=distance)
    return peaks


def _find_i0_platforms(
    x: np.array,
    min_c: float = 200,
    max_c: float = 250,
    min_len: int = 200,
    tail_point_num_for_std: int = 200,
    std_cutoff: float = 4.0
) -> np.array: #[[start, end],[start, end]]   [)
    """get I0 platform

    Args:
        x (np.array): signals
        min_c (float, optional): the lowest cutoff for I0. Defaults to 200.
        max_c (float, optional): the highest cutoff for I0. Defaults to 250.
        min_len (int, optional): the min len for I0 platform. Defaults to 200.
        tail_point_num_for_std (int, optional): std would be calculated based on tail this value points. Defaults to 200.
        std_cutoff (float, optional): the std of the tail `tail_point_num_for_std` points of a signal should be lower than this value
    Returns:
        np.array: [[start, end],[start, end]], each line represents a segment, coordinate system: 0-base, [)
    """
    if isinstance(x, list):
        x = np.array(x)
    candidate_pos = np.where((x>=min_c) & (x<=max_c))[0]
    index_x = np.zeros_like(x)
    index_x[candidate_pos] = 1
    break_points = np.where(np.diff(index_x)!=0)[0] + 1
    segments = np.concatenate(([0], break_points, [len(x)]))

    # filter segments
    filtered_segments = []
    for i in range(len(segments)-1):
        one_seg = x[segments[i]:segments[i+1]]
        if np.all(one_seg>min_c) and np.all(one_seg<=max_c) and len(one_seg)>=min_len and np.std(one_seg[-tail_point_num_for_std:])<std_cutoff:
            filtered_segments.append([segments[i],segments[i+1]])
    
    filtered_segments = np.array(filtered_segments, dtype=np.int32)

    return filtered_segments



def cal_read_num_for_each_type_each_channel(
    obj: dict,
):
    df = pd.DataFrame([[read_obj['type'], re.search(r'(channel\d+)', read_id).group(1)] for read_id, read_obj in obj.items()], columns=['type', 'channel'])
    read_num_df = df.groupby(['channel', 'type']).size().unstack()
    read_num_df = read_num_df.fillna(0)
    if 'type1' in read_num_df.columns:
        read_num_df['type1'] = read_num_df['type1'].astype('int32')
    if 'type2' in read_num_df.columns:
        read_num_df['type2'] = read_num_df['type2'].astype('int32')
    return read_num_df