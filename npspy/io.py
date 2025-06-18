#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: io.py
@Description: description of this file
@Datatime: 2025/01/14 11:37:46
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
import configparser
from configparser import ConfigParser, ExtendedInterpolation

from typing import Dict, Optional, Tuple, Union, Literal, List


def read_pickle(
    pickle_file_path: str,
) -> dict:
    """Read a pickled file and return a dict

    Args:
        pickle_file_path (str): the file path of pickled file

    Returns:
        dict: the content of pickled file
    """
    # data_dict = pickle.load(open(pickle_file_path, 'rb'))
    # return data_dict
    data_dict = {}
    with open(pickle_file_path, 'rb') as f:
        while True:
            try:
                chunk = pickle.load(f)
                data_dict.update(chunk)
            except EOFError:
                break
    return data_dict



def read_obj_or_pickle(
    obj: Union[dict, str],
) -> dict:
    """Read a pickled file or a dict obj. If is a dict obj, just copy it and return the copied one

    Args:
        obj (Union[dict, str]): a dict obj or a pickle file 

    Returns:
        dict: dict
    """
    if isinstance(obj, str):
        obj = read_pickle(obj)
    elif isinstance(obj, dict):
        obj = copy.deepcopy(obj)
    return obj

def save_pickle(
    obj,
    save_to_file: str,
    chunk_size: int = 500,
) -> None:
    """Save an obj to a pickle file.

    Args:
        obj (_type_): the obj to be saved
        save_to_file (str): the file name (with or with path) for saving obj
        chunk_size (int, optional): the chunk size for saving obj. Defaults to 500.
    """
    assert save_to_file[-4:] == ".pkl", "please provide file name ends with .pkl"
    save_dir = os.path.dirname(os.path.abspath(save_to_file))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # o = open(save_to_file, 'wb')
    # pickle.dump(obj, o)
    # o.close()

    # 将大字典分块保存到pickle文件
    keys = list(obj.keys())
    with open(save_to_file, 'wb') as f:
        for i in range(0, len(keys), chunk_size):
            chunk = {k: obj[k] for k in keys[i:i+chunk_size]}
            pickle.dump(chunk, f)




def save_np_as_npy(
    a: np.ndarray,
    npy_file_name: str
) -> None:
    """save a numpy array as a npy file

    Args:
        a (np.ndarray): the numpy array to be saved
        npy_file_name (str): the npy file
    """
    assert npy_file_name[-4:] == '.npy'
    with open(npy_file_name, "wb") as f:
        np.save(f, arr=a)

def load_npy_as_np(
    npy_file_name: str
) -> np.ndarray:
    """load a npy file to a numpy array

    Args:
        npy_file_name (str): the npy file

    Returns:
        np.ndarray: numpy array
    """
    assert npy_file_name[-4:] == '.npy'
    with open(npy_file_name, 'rb') as f:
        a = np.load(f)
    return a



def one_csv_file_to_dict(
    csv_file,
    tail: int = 0,
    label: str = None,
):
    df = pd.read_csv(csv_file)
    currents = df.iloc[-tail:,2].to_numpy()
    one_read_obj = {'signal': currents}
    if label != None:
        one_read_obj.update({'label': label})
    return one_read_obj


def read_csv_files_to_a_dict(
    csv_files: list,
    tail: int = 0,
    label: str = None,
) -> dict:
    """read csv files and save as one dict obj

    Args:
        csv_files (list): csv files
        tail (int, optional): select tail this number of signal. Defaults to 0.
                              0 means use all signal. 100 means use tail 100 signal
        label (str, optional): label. Defaults to None.

    Returns:
        dict: obj
    """
    obj = {}
    for one_csv_file in csv_files:
        read_id = os.path.basename(one_csv_file)
        read_id = re.search(r'(.*)\.csv$', read_id).group(1)
        obj[read_id] = one_csv_file_to_dict(one_csv_file,
                                            tail=tail,
                                            label=label)
    return obj


def read_dat_file(
    dat_files: Union[str, list],
    sort_dat_files: bool = True,
):
    if isinstance(dat_files, str):
        dat_files = [dat_files]

    if sort_dat_files:
        dat_files = _sort_dat_files(dat_files)
    
    signals = []
    for one_dat_file in dat_files:
        signals.append(np.fromfile(one_dat_file, dtype=np.float32))
    signals = np.concatenate(signals)
    return signals

def _sort_dat_files(
    dat_files: list,
) -> list:
    """return dat files with a specific order. the order are definded by the number after channelX

    Args:
        dat_files (list): dat_files list

    Returns:
        list: sorted dat files
    """
    dat_files = np.array(dat_files)
    dat_files = dat_files[np.argsort(np.array([int(re.search(r'channel\d+_(\d+)\.dat', one).group(1)) for one in dat_files]))]
    return list(dat_files)

def read_all_dat_files_in_a_channel(
    channel_path: str,
    sort_dat_files: bool = True,
):
    dat_files = glob.glob(f'{channel_path}/channel*dat')
    signals = read_dat_file(dat_files=dat_files, sort_dat_files=sort_dat_files)
    return signals


def read_ini_file_as_config(
    ini_file: str = 'config.ini',
):
    config = configparser.ConfigParser(interpolation=ExtendedInterpolation(),
                                       converters={'list': lambda x: [i.strip() for i in x.split(',')]},
                                       )
    config.read(ini_file)
    return config