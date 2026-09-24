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
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import copy
import torch

from .. import fast2pkl
from .. import tools as tl
from . import soft_dtw_cuda
from .. import io
from .. import machine_learning as ml
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

def find_degating_site_for_a_signal(
    signal: np.ndarray,
    degating_site_value_cut_off: float = -100.0,
    degating_site_distance_cut_off: int = 10000,
    degating_valley_len: int = 100,
    degating_valley_value_cut_off: float = -10.0,
) -> Tuple[List[int], Literal['single', 'multiple', 'no_degating']]:
    peak_indexs, _ = find_peaks(-signal, height=-degating_site_value_cut_off, distance=degating_site_distance_cut_off)
    degating_sites = []
    # print(peak_indexs)
    for peak_index in peak_indexs:
        if np.median(signal[peak_index:peak_index+degating_valley_len]) <= degating_valley_value_cut_off:
            degating_sites.append(peak_index)
    if len(degating_sites) == 1:
        return degating_sites, 'single'
    elif len(degating_sites) >= 2:
        return degating_sites,  'multiple'
    else:
        return [], 'no_degating'

def set_degating_site_for_an_obj(
    obj: dict,
    kwargs_for_find_degating_site_for_a_signal: Optional[dict] = None,
    in_place: bool = False,
) -> Optional[dict]:
    if not in_place:
        obj = copy.deepcopy(obj)

    kwargs_for_find_degating_site_for_a_signal = kwargs_for_find_degating_site_for_a_signal or {}
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'].astype(np.float32)
        degating_sites, degating_site_type = find_degating_site_for_a_signal(signal, **kwargs_for_find_degating_site_for_a_signal)
        read_obj['degating_sites'] = degating_sites
        read_obj['degating_site_type'] = degating_site_type
    
    if not in_place:
        return obj
    else:
        return None

def cut_degating_for_an_obj(
    obj: dict,
    end_len: int = 50,
    in_place: bool = False,
) -> Optional[dict]:
    if not in_place:
        obj = copy.deepcopy(obj)

    if 'degating_sites' not in obj[list(obj.keys())[0]]:
        set_degating_site_for_an_obj(obj, in_place=True)
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'].astype(np.float32)
        degating_sites = read_obj['degating_sites']
        if len(degating_sites) == 0:
            continue
        degating_site = min(degating_sites) - end_len
        degating_site = max(degating_site, 0)
        signal = signal[0:degating_site]
        read_obj['signal'] = signal
        read_obj['window'] = None

    if not in_place:
        return obj
    else:
        return None

def split_obj_by_degating_site_type(
    obj: dict,
):
    obj_degating = {}
    obj_no_degating = {}
    for read_id, read_obj in obj.items():
        degating_site_type = read_obj['degating_site_type']
        if degating_site_type == 'single':
            obj_degating[read_id] = read_obj
        elif degating_site_type == 'multiple':
            obj_degating[read_id] = read_obj
        elif degating_site_type == 'no_degating':
            obj_no_degating[read_id] = read_obj
    return obj_no_degating, obj_degating


def find_platform_start_of_the_end_platform_for_a_signal(
    signal: np.ndarray,
    size: int = 500,
    height_cut: int = 50,
):
    signal = signal.astype(np.float64)
    signal_median_filter = median_filter(signal, size=size)
    signal_median_filter_r = signal_median_filter[::-1]

    platform_start = len(signal_median_filter)
    for i in range(size, len(signal_median_filter_r)-size, size):
        pre_m = np.median(signal_median_filter_r[i-size:i])
        post_m = np.median(signal_median_filter_r[i+1:i+1+size])
        if pre_m - post_m >= height_cut:
            platform_start = i
            break
    platform_start = len(signal_median_filter) - platform_start
    return platform_start, signal_median_filter

def set_platform_start_of_the_end_platform_for_an_obj(
    obj: dict,
    in_place: bool = False,
) -> Optional[dict]:
    if not in_place:
        obj = copy.deepcopy(obj)

    for read_id, read_obj in obj.items():
        signal = read_obj['signal'].astype(np.float64)
        platform_start, signal_median_filter = find_platform_start_of_the_end_platform_for_a_signal(signal)
        read_obj['platform_start'] = platform_start
        # read_obj['platform_median'] = np.median(signal[platform_start:])
        read_obj['platform_std'] = np.std(signal[platform_start:])
        read_obj['platform_len'] = len(signal) - platform_start
    
    if not in_place:
        return obj
    else:
        return None
    
def define_trimming_end_platform_for_an_obj(
    obj: dict,
    platform_std_cutoff: float = 14.0,
    platform_start_cutoff: int = 1000,
    in_place: bool = False,
) -> Optional[dict]:
    if not in_place:
        obj = copy.deepcopy(obj)

    for read_id, read_obj in obj.items():
        read_obj['platform_need_trimming'] = False
        if read_obj['platform_std'] < platform_std_cutoff:
            if read_obj['platform_start'] > platform_start_cutoff:
                read_obj['platform_need_trimming'] = True
    
    if not in_place:
        return obj
    else:
        return None

def split_obj_by_platform_need_trimming(
    obj: dict,
):
    obj_need_trimming = {}
    obj_no_need_trimming = {}
    for read_id, read_obj in obj.items():
        if read_obj['platform_need_trimming']:
            obj_need_trimming[read_id] = read_obj
        else:
            obj_no_need_trimming[read_id] = read_obj
    return obj_no_need_trimming, obj_need_trimming

def cut_end_platform_for_an_obj(
    obj: dict,
    in_place: bool = False,
) -> Optional[dict]:
    if not in_place:
        obj = copy.deepcopy(obj)
    
    if 'platform_start' not in obj[list(obj.keys())[0]]:
        set_platform_start_of_the_end_platform_for_an_obj(obj, in_place=True)
    if 'platform_need_trimming' not in obj[list(obj.keys())[0]]:
        define_trimming_end_platform_for_an_obj(obj, in_place=True)
    
    for read_id, read_obj in obj.items():
        if read_obj['platform_need_trimming']:
            platform_start = read_obj['platform_start']
            read_obj['signal'] = read_obj['signal'][:platform_start]
    
    if not in_place:
        return obj
    else:
        return None
    
    





def find_platform_before_degating_site(
    signal: np.ndarray,
    kwargs_for_find_degating_site_for_a_signal: Optional[dict] = None,
    platform_extend_len: int = 500,
    # platform_std_cutoff: float = 10.0,
    platform_median_cutoff: float = 10.0,
    verbose: bool = False,
):
    signal = signal.astype(np.float32)
    kwargs_for_find_degating_site_for_a_signal = kwargs_for_find_degating_site_for_a_signal or {}
    degating_sites, degating_site_type = find_degating_site_for_a_signal(signal, **kwargs_for_find_degating_site_for_a_signal)
    if len(degating_sites) == 0:
        return 0, 'no_degating'
    elif len(degating_sites) >= 2:
        return degating_sites, 'multiple'
    else:
        degating_site = degating_sites[0]
        candidates = list(range(degating_site-100-platform_extend_len, 0, -platform_extend_len))
        for idx in range(2, len(candidates)):
            platform_start = candidates[idx]
            pre_one_platform_start = candidates[idx-1]
            pre_two_platform_start = candidates[idx-2]
            target_median = np.median(signal[platform_start:platform_start+platform_extend_len])
            pre_one_median = np.median(signal[pre_one_platform_start:pre_one_platform_start+platform_extend_len])
            pre_two_median = np.median(signal[pre_two_platform_start:pre_two_platform_start+platform_extend_len])
            if verbose:
                print(platform_start, pre_two_median, pre_one_median, target_median)
            if pre_one_median - target_median > platform_median_cutoff and pre_two_median - target_median > platform_median_cutoff:
                return platform_start, degating_site, 'single'
        return 0, degating_site, 'single'


        # for platform_start in range(degating_site-100-platform_extend_len, 0, -platform_extend_len):
        #     std_ = np.std(signal[platform_start:platform_start+platform_extend_len])
        #     if verbose:
        #         print(platform_start,std_)
        #     if std_ > platform_std_cutoff:
        #         return platform_start, degating_site, 'single'
        # return 0, degating_site, 'single'

def set_degating_info_for_an_obj(
    obj: dict,
):  
    degating_site_type_num_dict = {}
    for read_id, read_obj in obj.items():
        signal = read_obj['signal']
        res = find_platform_before_degating_site(signal)
        read_obj['degating_site_type'] = res[-1]
        degating_site_type_num_dict[res[-1]] = degating_site_type_num_dict.get(res[-1], 0) + 1
        if len(res) == 3:
            read_obj['platform_start'] = res[0]
            read_obj['degating_site'] = res[1]
        if res[-1] == 'multiple':
            read_obj['degating_sites'] = res[0]
    print(degating_site_type_num_dict)
            
            

        

        
def find_first_ap_for_a_signal(
    signal: np.ndarray, 
    openpore: float = 220.0, 
    offset: float = 0.4698, 
    start_trim_ratio: float = 0.3, 
    end_trim_ratio: float = 0.3,
    find_peak_min_peak_len: int = 15,
    find_peak_min_lagging_len: int = 200,

) -> Union[int, None]:
    y_cut = openpore * offset
    start_trim_len = round(start_trim_ratio * len(signal))
    end_trim_len = round(end_trim_ratio * len(signal))
    signal_trimmed = signal[start_trim_len:-end_trim_len]

    left = fast2pkl.find_peak(signal_trimmed, y_cut, min_peak_len=find_peak_min_peak_len, min_lagging_len=find_peak_min_lagging_len)
    if left is not None:
        left += start_trim_len
        if left > (1-end_trim_ratio) * len(signal):
            left = None
    
    return left

def find_second_ap_for_a_signal(signal, openpore=220.0, offset=0.4698, start_trim_ratio=0.5, end_trim_ratio=0.05,
             find_peak_min_peak_len: int = 15,
             find_peak_min_lagging_len: int = 200,

):
    y_cut = openpore * offset
    start_trim_len = round(start_trim_ratio * len(signal))
    end_trim_len = round(end_trim_ratio * len(signal))
    signal_trimmed = signal[start_trim_len:-end_trim_len]

    right = fast2pkl.find_peak(signal_trimmed[::-1], y_cut, min_peak_len=find_peak_min_peak_len, min_lagging_len=find_peak_min_lagging_len)
    if right is not None:
        right = len(signal) - end_trim_len - right

    return right

    
def find_windows_for_an_obj(
    obj: dict,
    in_place: bool = False,
    offset: float = 0.4698,
) -> Optional[dict]:
    if not in_place:
        obj = copy.deepcopy(obj)

    for read_id, read_obj in obj.items():
        left = find_first_ap_for_a_signal(read_obj['signal'], read_obj['OpenPore'], offset=offset)
        right = find_second_ap_for_a_signal(read_obj['signal'], read_obj['OpenPore'], offset=offset)
        if left is None:
            read_obj['first_ap'] = False
        elif left is not None:
            read_obj['first_ap'] = True

        if right is None:
            read_obj['second_ap'] = False
        elif right is not None:
            read_obj['second_ap'] = True

        if read_obj['first_ap']:
            if right is not None and right - left < 1000:
                right = None
            if right is None:
                right = len(read_obj['signal']) - 1
                read_obj['second_ap'] = False
            read_obj['window'] = (left, right)
        else:
            read_obj['window'] = None
    
    if not in_place:
        return obj
    else:
        return None
            

def extract_reads_with_window(obj):
    sub_obj = {}
    for read_id, read_obj in obj.items():
        if read_obj['window'] is not None:
            sub_obj[read_id] = read_obj
    return sub_obj


def extract_reads_with_only_first_ap(obj):
    sub_obj = {}
    for read_id, read_obj in obj.items():
        if read_obj['first_ap'] and not read_obj['second_ap']:
            sub_obj[read_id] = read_obj
    return sub_obj


def extract_reads_with_both_ap(obj):
    sub_obj = {}
    for read_id, read_obj in obj.items():
        if read_obj['first_ap'] and read_obj['second_ap']:
            sub_obj[read_id] = read_obj
    return sub_obj

def filter_by_dna1(
    obj: dict, 
    min_len: int = 5000, #6000
    max_len: int = 20000
) -> dict:
    valid_obj, unvalid_obj = {}, {}
    for read_id, read_obj in obj.items():
        s, e = read_obj['window']
        if s >= min_len and s <= max_len:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj

def get_dna1_length(obj):
    dna1_lens = []
    for read_id, read_obj in obj.items():
        s, e = read_obj['window']
        dna1_lens.append([read_id, s])
    dna1_lens = pd.DataFrame(dna1_lens, columns=['read_id', 'dna1_len'])
    return dna1_lens


def get_dna1_signal_for_dtw(
    obj: dict,
):
    dna1 = tl.get_signals_for_reads_in_an_obj(obj, down_sample_to=500, target='dna1', normalize_by_openpore=True)
    dna1_m = []
    for i in dna1:
        mf_s = median_filter(i, size=20)
        dna1_m.append(mf_s)
    dna1_m = np.array(dna1_m)
    dna1_m = dna1_m - dna1_m.mean(axis=1)[:,None]
    return dna1_m

def get_dna1_barycenter(
    obj: dict,
    seed: int = 42,
):
    from . import cluster
    ml.seed_everything(seed)
    dna1_m = get_dna1_signal_for_dtw(obj)
    series_list = torch.Tensor(dna1_m)[:,:,None].cuda()
    init_center = series_list.mean(axis=0)[None,:,:].cuda()
    sdtw = soft_dtw_cuda.SoftDTW(use_cuda=True, gamma=0.01, normalize=False)
    barycenter = cluster.compute_soft_dtw_barycenter(series_list, init_center, sdtw, batch_size=1000, lr=0.01, n_iterations=30)
    return barycenter

def save_barycenter(
    barycenter: torch.Tensor,
    save_dir: str = './',
    save_name: str = 'dna1_barycenter.npy',
):
    io.save_np_as_npy(barycenter.cpu().numpy(), save_dir, save_name)

def load_barycenter(
    file_path: str,
):
    barycenter = torch.from_numpy(io.load_npy_as_np(file_path)).cuda()
    return barycenter


def cal_distance_to_barycenter(
    obj: dict,
    barycenter: torch.Tensor,
    type: Literal['dna1', 'dna2'] = 'dna1',
) -> None:
    read_ids = list(obj.keys())
    if type == 'dna1':
        dna_m = get_dna1_signal_for_dtw(obj)
    elif type == 'dna2':
        dna_m = get_dna2_signal_for_dtw(obj)
    sdtw = soft_dtw_cuda.SoftDTW(use_cuda=True, gamma=0.01, normalize=False)
    distances = []
    for i in range(0, len(read_ids), 2000):
        sub_data = torch.Tensor(dna_m[i:i+2000])[:,:,None].cuda()
        distance = sdtw(sub_data, barycenter.expand(len(sub_data), -1, -1).cuda()).cpu().numpy()
        distances.extend(distance)
    for read_id, dis in zip(read_ids, distances):
        obj[read_id][f'distance_to_{type}_barycenter'] = dis
    
def filter_by_dna1_shape(
    obj: dict,
    distance_cutoff: float = -7.0,
):
    valid_obj, unvalid_obj = {}, {}
    for read_id, read_obj in obj.items():
        if read_obj['distance_to_dna1_barycenter'] is not None and read_obj['distance_to_dna1_barycenter'] < distance_cutoff:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj

def get_dna2_length(obj):
    dna2_lens = []
    for read_id, read_obj in obj.items():
        s, e = read_obj['window']
        total_len = len(read_obj['signal'])
        dna2_lens.append([read_id, total_len-e])
    dna2_lens = pd.DataFrame(dna2_lens, columns=['read_id', 'dna2_len'])
    return dna2_lens


def get_reads_with_second_ap(
    obj: dict,
    min_dna2_len: int = 1000,
):
    sub_obj = {}
    for read_id, read_obj in obj.items():
        s, e = read_obj['window']
        if e - s < min_dna2_len:
            continue
        if read_obj['second_ap']:
            sub_obj[read_id] = read_obj
    return sub_obj

def get_dna2_signal_for_dtw(
    obj: dict,
):
    dna2 = tl.get_signals_for_reads_in_an_obj(obj, down_sample_to=500, target='dna2', normalize_by_openpore=True)
    dna2_m = []
    for i in dna2:
        mf_s = median_filter(i, size=20)
        dna2_m.append(mf_s)
    dna2_m = np.array(dna2_m)
    dna2_m = dna2_m - dna2_m.mean(axis=1)[:,None]
    return dna2_m

def get_dna2_barycenter(
    obj: dict,
    seed: int = 42,
):
    from . import cluster
    ml.seed_everything(seed)
    dna2_m = get_dna2_signal_for_dtw(obj)
    series_list = torch.Tensor(dna2_m)[:,:,None].cuda()
    init_center = series_list.mean(axis=0)[None,:,:].cuda()
    sdtw = soft_dtw_cuda.SoftDTW(use_cuda=True, gamma=0.01, normalize=False)
    barycenter = cluster.compute_soft_dtw_barycenter(series_list, init_center, sdtw, batch_size=1000, lr=0.01, n_iterations=30)
    return barycenter

def filter_by_dna2_shape(
    obj: dict,
    distance_cutoff: float = -7.0,
):
    valid_obj, unvalid_obj = {}, {}
    for read_id, read_obj in obj.items():
        if read_obj['distance_to_dna2_barycenter'] is not None and read_obj['distance_to_dna2_barycenter'] < distance_cutoff:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj


def filter_by_dna1_I2I0(
    obj: dict,
    min_I2I0: float = 0.31,
    max_I2I0: float = 0.43,
):
    obj = tl.set_att_for_an_obj(obj, atts=['mean_of_I/I0_of_dna1'])
    valid_obj, unvalid_obj = {}, {}
    for read_id, read_obj in obj.items():
        if read_obj['mean_of_I/I0_of_dna1'] is not None and min_I2I0 <= read_obj['mean_of_I/I0_of_dna1'] <= max_I2I0:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj

def cal_ap1_higher_ratio(obj):
    higher_ap1_ratio = []
    for read_id, read_obj in obj.items():
        s = read_obj['window'][0]
        signal = read_obj['signal'][s//2:s].astype(np.float32)
        cutoff = read_obj['signal'][s]
        ratio = np.mean(signal >= cutoff)
        read_obj['ap1_ratio'] = ratio
        higher_ap1_ratio.append([read_id, ratio])
    return pd.DataFrame(higher_ap1_ratio, columns=['read_id', 'ratio'])

def filter_by_ap1_len(
    obj, 
    ap1_ratio_min_cutoff: float = 0.03,
    ap1_ratio_max_cutoff: float = 0.32
) -> dict:
    valid_obj, unvalid_obj = {}, {}
    higher_ap1_ratio = cal_ap1_higher_ratio(obj)
    df = higher_ap1_ratio.set_index('read_id')
    for read_id, read_obj in obj.items():
        if df.loc[read_id, 'ratio'] >= ap1_ratio_min_cutoff and df.loc[read_id, 'ratio'] <= ap1_ratio_max_cutoff:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj

def cal_ap2_higher_ratio(obj):
    higher_ap2_ratio = []
    for read_id, read_obj in obj.items():
        if not read_obj['second_ap']:
            continue
        e = read_obj['window'][1]
        total_len = len(read_obj['signal'])
        signal = read_obj['signal'][e:(e+(total_len-e)//2)].astype(np.float32)
        cutoff = read_obj['signal'][e]
        # if len(signal) == 0:
        #     continue
        ratio = np.mean(signal >= cutoff)
        read_obj['ap2_ratio'] = ratio
        higher_ap2_ratio.append([read_id, ratio])
    return pd.DataFrame(higher_ap2_ratio, columns=['read_id', 'ratio'])

def filter_by_ap2_len(obj, min_ratio=0.05, max_ratio=0.70):
    valid_obj, unvalid_obj = {}, {}
    higher_ap2_ratio = cal_ap2_higher_ratio(obj)
    df = higher_ap2_ratio.set_index('read_id')
    for read_id, read_obj in obj.items():
        if not read_obj['second_ap']:
            valid_obj[read_id] = read_obj
            continue
        # if read_id not in set(df.index):
        #     new_obj[read_id] = read_obj
        #     continue
        if df.loc[read_id, 'ratio'] >= min_ratio and df.loc[read_id, 'ratio'] <= max_ratio:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj

def cal_std_for_target_signals_in_an_obj(obj):
    stat_df = []
    for read_id, read_obj in obj.items():
        s, e = read_obj['window']
        signal = read_obj['signal'][s:e].astype(np.float32)
        std_ = np.std(signal)
        stat_df.append([read_id, std_])
        read_obj['std'] = std_
    stat_df = pd.DataFrame(stat_df, columns=['read_id', 'std'])
    return stat_df


def filter_by_std_of_window_signal_for_an_obj(obj, min_std=10.0, max_std=40.0):
    stat_df = cal_std_for_target_signals_in_an_obj(obj)
    stat_df = stat_df.set_index('read_id')
    valid_obj, unvalid_obj = {}, {}
    for read_id, read_obj in obj.items():
        s, e = read_obj['window']
        std_ = stat_df.loc[read_id, 'std']
        if std_ >= min_std and std_ <= max_std:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj
    

def filter_by_polyT(obj):
    valid_obj, unvalid_obj = {}, {}
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'].astype(np.float32)
        start, end = read_obj['window']
        signal = signal[start:end]
        polyT_signal = signal[200:1000]
        polyT_std = np.std(polyT_signal)
        read_obj['polyT_i2io'] = np.mean(polyT_signal / read_obj['OpenPore'])
        read_obj['polyT_std'] = polyT_std
        if polyT_std < 6:
            valid_obj[read_id] = read_obj
        else:
            unvalid_obj[read_id] = read_obj
    return valid_obj, unvalid_obj
    

def get_signals_for_dtw(obj, norm_by_polyT: bool = False, norm_by_polyT_as: float = 0.3, down_sample_to: int = 200):
    signals = []
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'] / read_obj['OpenPore']
        s, e = read_obj['window']
        signal = tl.down_sampling(signal[s:e], down_sample_to=down_sample_to)
        if norm_by_polyT:
            polyT_i2io = read_obj['polyT_i2io']
            signal = signal / polyT_i2io * norm_by_polyT_as
        signals.append(signal[None,:])  
    signals = np.concatenate(signals)
    return signals


def get_polyT_normed_signals_fom_an_obj(obj, norm_by_polyT_as: float = 0.3):
    signals = []
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'] / read_obj['OpenPore']
        s, e = read_obj['window']
        polyT_i2io = read_obj['polyT_i2io']
        signal = signal / polyT_i2io * norm_by_polyT_as
        signals.append(signal)
    return signals


def get_percentile_for_signals(
    signals: list[np.ndarray],
):
    features = []
    for signal in signals:
        features_for_one_signal = []
        features_for_one_signal.extend([np.percentile(signal, per) for per in np.arange(0, 100, 5)])
        # features_for_one_signal.extend([np.mean(signal), np.std(signal)])
        features.append(features_for_one_signal)
    return np.array(features, dtype=np.float32)
            