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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import os

from . import tools as tl
from .. import plot as pl

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

def get_lh_stat(obj, norm_by_polyT_as: float = 0.3, min_cut=0.22, max_cut=0.38):
    stat_df = []
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'].astype(np.float32) / read_obj['OpenPore']
        start, end = read_obj['window']
        # if end is None:
        #     signal = signal[start:]
            
        # else:
        #     signal = signal[start:end]

        signal = signal[start:end]
    
        signal = tl.trim_array_percentiles(signal)

        polyT_i2io = read_obj['polyT_i2io']
        signal = signal / polyT_i2io * norm_by_polyT_as        
        
        low_length = max(1, np.sum(signal<=min_cut))
        low_mean = np.mean(signal[signal<=min_cut]) if signal[signal<=min_cut].size > 0 else min_cut
        high_length = max(1, np.sum(signal>=max_cut))
        high_mean = np.mean(signal[signal>=max_cut]) if signal[signal>=max_cut].size > 0 else max_cut
        total_length = len(signal)
        stat_df.append([read_id, low_length, high_length, total_length, low_mean, high_mean])
    stat_df = pd.DataFrame(stat_df, columns=['read_id', 'low_len', 'high_len', 'total_length', 'low_mean', 'high_mean'])
    stat_df['l2h'] = stat_df['low_len'] / stat_df['high_len']
    stat_df['log1p_l2h'] = np.log1p(stat_df['l2h'])
    stat_df['low2lowhigh'] = stat_df['low_len'] / (stat_df['low_len'] + stat_df['high_len'])
    stat_df['low2total'] = stat_df['low_len'] / stat_df['total_length']
    stat_df['high2total'] = stat_df['high_len'] / stat_df['total_length']
    stat_df['middle_ratio'] = (stat_df['total_length'] - stat_df['low_len'] - stat_df['high_len']) / stat_df['total_length']
    stat_df['low_area'] = stat_df['low_mean'] * (stat_df['low_len'] / stat_df['total_length'])
    stat_df['high_area'] = stat_df['high_mean'] * (stat_df['high_len'] / stat_df['total_length'])
    return stat_df

def get_bind_stat(obj, min_cut, max_cut, log1p_l2h_cutoff):
    stat_df = get_lh_stat(obj, min_cut=min_cut, max_cut=max_cut)
    stat_df['bind'] = 0
    stat_df.loc[stat_df['log1p_l2h']>log1p_l2h_cutoff, 'bind'] = 1
    unbind_num = np.sum(stat_df['bind'] == 0)
    bind_num = np.sum(stat_df['bind'] == 1)
    bind_ratio = bind_num / (unbind_num + bind_num)
    return stat_df, unbind_num, bind_num, bind_ratio

def train_bind_stat_using_an_obj_by_kmeans(
    obj: dict,
    min_cut: float,
    max_cut: float,
    seed: int = 42,
    save_dir: str = './',
    save_file_name: str = 'kmeans_pipeline.pkl',
):
    stat_df = get_lh_stat(obj, min_cut=min_cut, max_cut=max_cut)
    X = stat_df[['low2total', 'high2total']]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, random_state=seed))
    ])
    pipeline.fit(X)
    pl.create_dir_if_not_exist(save_dir)
    joblib.dump(pipeline, os.path.join(save_dir, save_file_name))
    
    return pipeline

def get_bind_stat_for_an_obj_by_kmeans(
    obj: dict,
    pipeline_path: str,
    min_cut: float,
    max_cut: float,
):
    pipeline = joblib.load(pipeline_path)
    stat_df = get_lh_stat(obj, min_cut=min_cut, max_cut=max_cut)
    X = stat_df[['low2total', 'high2total']]
    clusters = pipeline.predict(X)
    if pipeline['kmeans'].cluster_centers_[0,0] > pipeline['kmeans'].cluster_centers_[1,0]:
        mapping = {0: 'bind', 1: 'unbind'}
    else:
        mapping = {0: 'unbind', 1: 'bind'}
    clusters = np.array([mapping[i] for i in clusters])
    stat_df['kmeans_cluster'] = clusters
    return stat_df






