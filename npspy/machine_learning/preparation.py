#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: preparation.py
@Description: description of this file
@Datatime: 2025/01/20 09:55:18
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
'''

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Literal, List
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .. import tools as tl
from .. import io

def set_y_codes_for_classes(
    class_labels: List[Union[str, int]],
) -> Dict:
    """_summary_

    Args:
        class_labels (List[Union[str, int]]): a two dimentional list, each element is a class and contains a list of amino acids.

    Returns:
        Dict: y code dict
    """

    y_codes = {}
    for class_label, one_class_aas in enumerate(class_labels):
        for aa in one_class_aas:
            y_codes[aa] = class_label
    return y_codes


def get_X_y_from_an_obj(
    obj: List[Union[dict, str]],
    y_code_dict: dict,
    label: str,
    att: str = 'lowpass_signal',
    down_sample_to: int = 100,
) -> pd.DataFrame:
    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    X = tl.get_signals_for_reads_in_an_obj(obj, att=att, down_sample_to=down_sample_to)
    y = [y_code_dict[label]] * X.shape[0]
    X = X.astype(np.float32)
    y = np.array(y)
    read_ids = list(obj.keys())
    df = pd.DataFrame(np.concatenate([X, y[:,None]], axis=1), index=read_ids)
    df.columns = [i for i in range(df.shape[1]-1)] + ['y']
    return df

def get_X_y_from_objs(
    objs: List[Union[dict, str]],
    labels: List[str],
    y_code_dict: dict,
    att: str = 'lowpass_signal',
    down_sample_to: int = 100,
) -> pd.DataFrame:
    df = []
    for obj, label in zip(objs, labels):
        if isinstance(obj, str):
            obj = io.read_pickle(obj)
        df_ = get_X_y_from_an_obj(obj, y_code_dict, label=label, att=att, down_sample_to=down_sample_to)
        df.append(df_)
    df = pd.concat(df)
    return df

def select_the_same_sample_num_for_all_class(
    df: pd.DataFrame,
    sample_num: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    if sample_num is None:
        sample_num = df.groupby('y').size().min()
    
    df = df.groupby('y').sample(n=sample_num, random_state=seed)
    return df

class Cus_Dataset(Dataset):
    def __init__(self, data_df, augment: bool = False):
        self.all_read_ids = data_df.index.to_numpy()
        self.X = data_df.iloc[:,:-1].to_numpy(dtype=np.float32)
        if 'y' in data_df.columns:
            self.y = data_df['y'].to_numpy(dtype=np.int32)
        else:
            self.y = np.zeros(len(data_df), dtype=np.int32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_read_id = self.all_read_ids[idx]
        one_X = self.X[idx]
        one_y = self.y[idx]

        if self.augment:
            one_X = window_warping(one_X)

        one_x = torch.as_tensor(one_y, dtype=torch.float32)
        one_y = torch.as_tensor(one_y, dtype=torch.long)

        return one_read_id, one_X, one_y
    
def construct_dataloader_from_data_df(
    data_df: pd.DataFrame,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    augment: bool = False,
) -> DataLoader:
    dataset = Cus_Dataset(data_df, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def window_warping(data):
    seq_length = len(data)
    warp_width = 20
    warp_factor = np.random.uniform(low=0.0, high=2)
    if warp_factor <= 1:
        return data

    warp_start = np.random.randint(0, seq_length - warp_width)
    warp_end = warp_start + warp_width

    # 创建时间索引
    time_index = np.arange(seq_length)
    warp_index = np.linspace(warp_start, warp_end, int(warp_width * warp_factor))

    # 替换时间索引
    time_index = np.concatenate([time_index[0:warp_start], warp_index, time_index[warp_end+1:]])
    warped_data = np.interp(time_index, np.arange(seq_length), data)

    warped_data = resize(warped_data, seq_length)
    warped_data = warped_data.astype(np.float32)
    return warped_data


def resize(arr, new_len):
    if len(arr.shape) > 1:
        arr = np.squeeze(arr)
    length = len(arr)
    indices_new = np.linspace(0, length-1, new_len)
    indices_old = np.arange(0, length)
    return np.interp(indices_new, indices_old, arr)