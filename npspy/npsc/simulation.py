#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: simulation.py
@Description: description of this file
@Datatime: 2025/07/04 10:58:04
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy import signal
import re

from .. import io
from .. import machine_learning as ml
from . import tools as thistl


def random_nonuniform_scale(
    sequence, 
    num_segments=10,
    max_scale=2.0
) -> np.ndarray:
    """Generate a non-uniformly scaled version of a sequence. The sequence is scaled by random factors in different segments.

    Args:
        sequence (np.ndarray): The input sequence to be scaled.
        num_segments (int): The number of segments to divide the sequence into for scaling.
        max_scale (float): The maximum scaling factor for any segment.
    Returns:
        np.ndarray: The scaled sequence.
    """

    n = len(sequence)
    original_t = np.linspace(0, 1, n)
    
    # 1. 生成随机缩放因子(分段)
    segment_points = np.sort(np.random.randint(0, n, num_segments-1))
    segment_points = np.concatenate([[0], segment_points, [n]])
    scale_factors = np.random.uniform(1/max_scale, max_scale, num_segments)
    
    # 2. 构建非均匀时间映射
    scaled_t = np.zeros(n)
    for i in range(num_segments):
        start, end = segment_points[i], segment_points[i+1]
        segment_len = end - start
        scaled_segment_len = int(segment_len * scale_factors[i])
        scaled_t[start:end] = np.linspace(
            scaled_t[start-1] if i>0 else 0,
            scaled_t[start-1]+scaled_segment_len/n if i>0 else scaled_segment_len/n,
            segment_len
        )
    
    # 3. 归一化时间轴并插值
    scaled_t = scaled_t / scaled_t.max()
    cs = CubicSpline(original_t, sequence)
    scaled_sequence = cs(scaled_t)
    
    return scaled_sequence


def generate_time_series_situation1(
    num_sequences: int = 20000,
    seq_length: int = 1000, 
    signal_std: float = 0.02,
    noise_ratio: float = 0.1,
    noisy_signal_std: float = 0.04,
    seed: int = 41,
):
    ml.seed_everything(seed)
    # 计算目标序列和噪音序列数量
    target_count = int(num_sequences * (1 - noise_ratio))
    
    # 初始化存储
    sequences = np.zeros((num_sequences, seq_length))
    labels = np.zeros(num_sequences, dtype='<U6')  # 'target'或'noise'
    labels = labels.astype(str)
    
    # 生成时间轴
    t = np.linspace(0, 2*np.pi, seq_length)
    
    # 生成正弦目标序列
    for i in range(target_count):
        # phase = np.random.uniform(0, 2*np.pi)
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'target'
    
    # 生成噪音比较大的正弦序列
    for i in range(target_count, num_sequences):
        # phase = np.random.uniform(0, 2*np.pi)
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0,  noisy_signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'noise'
    
    # 打乱顺序
    indices = np.random.permutation(num_sequences)
    return sequences[indices], labels[indices]
    

def generate_time_series_situation2(
    num_sequences: int = 20000,
    seq_length: int = 1000, 
    signal_std: float = 0.02,
    noise_ratio: float = 0.1,
    seed: int = 41,
):
    ml.seed_everything(seed)

    # 计算目标序列和噪音序列数量
    target_count = int(num_sequences * (1 - noise_ratio))
    
    # 初始化存储
    sequences = np.zeros((num_sequences, seq_length))
    labels = np.zeros(num_sequences, dtype='<U6')  # 'target'或'noise'
    labels = labels.astype(str)
    
    # 生成时间轴
    t = np.linspace(0, 2*np.pi, seq_length)
    
    # 生成正弦目标序列
    for i in range(target_count):
        # phase = np.random.uniform(0, 2*np.pi)
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'target'

    # 生成时间轴
    t = np.linspace(0, 3.5*np.pi, seq_length)
    
    # 生成方波噪音序列 并上下flip
    for i in range(target_count, num_sequences):
        # phase = np.random.uniform(0, 2*np.pi)
        phase = 0
        sequences[i] = ((signal.square(t + phase) * 0.8) + 1) / 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        sequences[i] = 1 - sequences[i]
        labels[i] = 'noise'

    # 打乱顺序
    indices = np.random.permutation(num_sequences)
    return sequences[indices], labels[indices]


def generate_time_series_situation3(
    num_sequences: int = 20000,
    seq_length: int = 1000, 
    signal_std: float = 0.02,
    noise_ratio: float = 0.1,
    seed: int = 41,
):
    ml.seed_everything(seed)

    # 计算目标序列和噪音序列数量
    target_count = int(num_sequences * (1 - noise_ratio))
    
    # 初始化存储
    sequences = np.zeros((num_sequences, seq_length))
    labels = np.zeros(num_sequences, dtype='<U6')  # 'target'或'noise'
    labels = labels.astype(str)
    
    # 生成时间轴
    t = np.linspace(0, 2*np.pi, seq_length)
    
    # 生成正弦目标序列
    for i in range(target_count):
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'target'


    # 生成正弦噪音序列，两个周期
    t = np.linspace(0, 4*np.pi, seq_length)
    for i in range(target_count, num_sequences):
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'noise'

    # 打乱顺序
    indices = np.random.permutation(num_sequences)
    return sequences[indices], labels[indices]


def generate_time_series_situation4(
    num_sequences: int = 20000,
    seq_length: int = 1000, 
    signal_std: float = 0.02,
    noise_ratio: float = 0.1,
    seed: int = 41,
):
    ml.seed_everything(seed)

    # 计算目标序列和噪音序列数量
    target_count = int(num_sequences * (1 - noise_ratio))
    
    # 初始化存储
    sequences = np.zeros((num_sequences, seq_length))
    labels = np.zeros(num_sequences, dtype='<U6')  # 'target'或'noise'
    labels = labels.astype(str)
    
    # 生成时间轴
    t = np.linspace(0, 2*np.pi, seq_length)
    
    # 生成正弦目标序列
    for i in range(target_count):
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'target'


    # 生成直线噪音序列
    for i in range(target_count, num_sequences):
        phase = 0
        sequences[i] = np.array([0.5] * seq_length ) # 0.5
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'noise'

    # 打乱顺序
    indices = np.random.permutation(num_sequences)
    return sequences[indices], labels[indices]


def generate_time_series_situation5(
    num_sequences: int = 20000,
    seq_length: int = 1000, 
    signal_std: float = 0.02,
    noise_ratio: float = 0.1,
    seed: int = 41,
    noisy_signal_std: float = 0.04,
):
    ml.seed_everything(seed)

    # 计算目标序列和噪音序列数量
    target_count = int(num_sequences * (1 - noise_ratio))
    noise_count = int(num_sequences * noise_ratio)
    
    # 初始化存储
    sequences = np.zeros((num_sequences, seq_length))
    labels = np.zeros(num_sequences, dtype='<U6')  # 'target'或'noise'
    labels = labels.astype(str)
    
    # 生成时间轴
    t = np.linspace(0, 2*np.pi, seq_length)
    
    # 生成正弦目标序列
    for i in range(target_count):
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'target'


    # 生成噪音比较大的正弦序列
    for i in range(target_count, target_count + noise_count//2):
        # phase = np.random.uniform(0, 2*np.pi)
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0,  noisy_signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'noise'
    
    # 生成正弦噪音序列，两个周期
    t = np.linspace(0, 4*np.pi, seq_length)
    for i in range(target_count + noise_count//2, num_sequences):
        phase = 0
        sequences[i] = ((np.sin(t + phase) * 0.8) + 1 )/ 2  # 归一化到0.1-0.9
        sequences[i] += np.random.normal(0, signal_std, seq_length)
        sequences[i] = random_nonuniform_scale(sequences[i])
        sequences[i] = np.clip(sequences[i], 0, 1)
        labels[i] = 'noise'

    # 打乱顺序
    indices = np.random.permutation(num_sequences)
    return sequences[indices], labels[indices]


def generate_time_series_as_obj(
    situation: Literal['situation1', 'situation2', 'situation3', 'situation4'],
    save_name: str,
    save_dir: str = '.',
    seed: int = 41,
    
):
    if situation == 'situation1':
        X, y = generate_time_series_situation1(seed=seed)
    elif situation == 'situation2':
        X, y = generate_time_series_situation2(seed=seed)
    elif situation == 'situation3':
        X, y = generate_time_series_situation3(seed=seed)
    elif situation == 'situation4':
        X, y = generate_time_series_situation4(seed=seed)
    elif situation == 'situation5':
        X, y = generate_time_series_situation5(seed=seed)

    obj = {}
    for indx, (one_X, one_y) in enumerate(zip(X,y)):
        read_id = f'read_{indx}_{one_y}'
        obj[read_id] = {
            'signal': one_X,
            'window': (0,1000),
            'OpenPore': 1.0,
            'label': one_y
        }
    io.save_pickle(obj, f'{save_dir}/{save_name}.pkl')



def get_pred_from_errors_for_sim(
    error_df: Union[pd.DataFrame, str],
):
    if isinstance(error_df, str):
        error_df = pd.read_csv(error_df, index_col=0)
    
    error_df['true'] = [re.search(r'.*_(\S+)$', i).group(1) for i in error_df.index]

    noise_read_num = np.sum(error_df['true'] == 'noise')
    error_df['pred'] = 'target'
    error_df.loc[error_df.sort_values('reconstruction_error').tail(noise_read_num).index, 'pred'] = 'noise'
    
    pred_df = error_df[['true', 'pred']].copy()
    return pred_df


def get_cm_from_errors_for_sim(
    error_df: Union[pd.DataFrame, str],
):
    pred_df = get_pred_from_errors_for_sim(error_df)
    cm_df = ml.get_cm(pred_df, label_order=['target', 'noise'])
    return cm_df

def get_metrics_from_errors_for_sim(
    error_df: Union[pd.DataFrame, str],
):
    pred_df = get_pred_from_errors_for_sim(error_df)
    metrics_df = thistl.get_metrics_from_pred_df(pred_df)
    return metrics_df
