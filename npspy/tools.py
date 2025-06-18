import random
import numpy as np
import pandas as pd
import copy
from scipy import interpolate
from scipy.ndimage import median_filter
from typing import Dict, Optional, Tuple, Union, Literal, List
from scipy.signal import butter, lfilter, find_peaks
from pathlib import Path

from . import preprocessing as pp
from . import io

def extract_reads_as_an_obj(
    obj: dict,
    read_ids: list,
) -> dict:
    """extract reads in an obj as a new obj

    Args:
        obj (dict): obj
        read_ids (list): read ids for reads that would be selected

    Returns:
        dict: new obj
    """
    sub_obj = {}
    for read_id in read_ids:
        sub_obj[read_id] = copy.deepcopy(obj[read_id])
    return sub_obj

def delete_reads_in_an_obj(
    obj: dict,
    reads_need_to_remove: list,
    in_place: bool = False,
) -> Union[dict, None]:
    
    if in_place:
        sub_obj = obj
    else:
        sub_obj = copy.deepcopy(obj)

    for read_id in reads_need_to_remove:
        del sub_obj[read_id]

    if in_place:
        return None
    else:
        return sub_obj

    
    

def extract_reads_with_window(
    obj: dict,
) -> dict:
    read_ids = []
    for read_id, read_obj in obj.items():
        if 'window' in read_obj and read_obj['window'] != None:
            read_ids.append(read_id)
    obj = extract_reads_as_an_obj(obj, read_ids=read_ids)
    return obj



def extract_reads_with_labels(
    obj: dict,
    labels: Union[list, str],
) -> dict:
    """extract reads with labels

    Args:
        obj (dict): obj
        labels (list): labels

    Returns:
        dict: obj
    """
    if isinstance(labels, str):
        labels = [labels]
    obj = {read_id:read_obj for read_id, read_obj in obj.items() if read_obj['label'] in labels}
    return obj

def extract_reads_by_stair(
    obj: dict,
    stair_nums: Union[int, list],
    read_num: int = -1,
    seed: int = 0,
) -> dict:
    read_ids = []
    if isinstance(stair_nums, int):
        stair_nums = [stair_nums]
    for read_id, read_obj in obj.items():
        if len(read_obj['transitions']) - 1 in stair_nums:
            read_ids.append(read_id)
    if read_num > 0:
        np.random.seed(seed)
        np.random.shuffle(read_ids)
        read_ids = read_ids[:read_num]
    obj = extract_reads_as_an_obj(obj, read_ids=read_ids)
    return obj
    


def substrac_signal_of_an_obj(
    obj: dict,
    tail: int = 0,
    in_place: bool = False,
):
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        read_obj['signal'] = read_obj['signal'][-tail:]
    
    if in_place:
        return None
    else:
        return new_obj
    

def set_att_for_an_obj(
    obj: dict,
    atts: Literal['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length', 'pd2rd', 'signal_length', 'pd2od', 'dna1_len', 'dna2_len', 'min_of_I/I0'] = ['mean_of_I/I0', 'std_of_I/I0', 'window_length'],
    in_place: bool = False,
    scale_by_openpore: bool = True,
) -> Union[dict, None]:
    assert np.all(np.isin(atts, ['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length', 'pd2rd', 'signal_length', 'pd2od', 'dna1_len', 'dna2_len', 'min_of_I/I0'])) == True

    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        if 'window' in read_obj and read_obj['window'] != None:
            x = read_obj['signal'][read_obj['window'][0]:read_obj['window'][1]]
        else:
            x = read_obj['signal']
        
        x = x.astype(np.float32)

        if scale_by_openpore:
            x = x/read_obj['OpenPore']

        if 'mean_of_I/I0' in atts:
            read_obj['mean_of_I/I0'] = np.mean(x)
        if 'std_of_I/I0' in atts:
            read_obj['std_of_I/I0'] = np.std(x)
        if 'median_of_I/I0' in atts:
            read_obj['median_of_I/I0'] = np.median(x)
        if 'min_of_I/I0' in atts:
            read_obj['min_of_I/I0'] = np.min(x)
        if 'window_length' in atts:
            if 'window' in read_obj and read_obj['window'] != None:
                read_obj['window_length'] = read_obj['window'][1] - read_obj['window'][0]
            else:
                read_obj['window_length'] = None
        if 'dna1_len' in atts:
            if 'window' in read_obj and read_obj['window'] != None:
                s, e = read_obj['window'][0], read_obj['window'][1]
                read_obj['dna1_len'] = s
            else:
                read_obj['dna1_len'] = None
        if 'dna2_len' in atts:
            if 'window' in read_obj and read_obj['window'] != None:
                s, e = read_obj['window'][0], read_obj['window'][1]
                read_obj['dna2_len'] = len(read_obj['signal']) - e
            else:
                read_obj['dna2_len'] = None
        if 'pd2rd' in atts:
            window_len = len(read_obj['signal'])
            if 'window' in read_obj and read_obj['window'] != None:
                window_len = read_obj['window'][1] - read_obj['window'][0]
            read_obj['pd2rd'] = window_len/len(read_obj['signal'])
        if 'signal_length' in atts:
            read_obj['signal_length'] = len(read_obj['signal'])
        if 'pd2od' in atts:
            window_len = len(read_obj['signal'])
            if 'window' in read_obj and read_obj['window'] != None:
                window_len = read_obj['window'][1] - read_obj['window'][0]
            read_obj['pd2od'] = window_len/(len(read_obj['signal']) - window_len)

    if in_place:
        return None
    else:
        return new_obj

def get_att_from_an_obj(
    obj: dict,
    atts: Literal['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length', 'pd2rd', 'signal_length', 'pd2od', 'dna1_len', 'dna2_len']
):
    assert np.all(np.isin(atts, ['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length', 'pd2rd', 'signal_length', 'pd2od', 'dna1_len', 'dna2_len'])) == True
    
    atts_df = []
    read_ids = []
    for read_id, read_obj in obj.items():
        atts_df.append([read_obj[att] for att in atts])
        read_ids.append(read_id)
    atts_df = pd.DataFrame(atts_df, columns=atts)
    atts_df.index = read_ids
    return atts_df
    
def filter_out_reads_without_widows(
    obj,
):
    """Filter out the reads with no target windows.

    Args:
        obj (_type_): the obj to be filtered
    """
    read_ids_with_no_windows = [read_id for read_id, read_obj in obj.items() if read_obj['window']==None]
    obj = delete_reads_in_an_obj(obj=obj, reads_need_to_remove=read_ids_with_no_windows)
    return obj

def filter_out_reads_with_negative_signals_in_window(
    obj,
):
    read_ids_with_negtive_signals = [read_id for read_id, read_obj in obj.items() if np.any(read_obj['signal'][read_obj['window'][0]:read_obj['window'][1]]<0)]
    obj = delete_reads_in_an_obj(obj=obj,  reads_need_to_remove=read_ids_with_negtive_signals)
    return obj

def filter_out_reads_with_curr_high_than_i0_in_window(
    obj: dict,
):
    read_ids_with_signal_higher_than_i0 = [read_id for read_id, read_obj in obj.items() if np.any(read_obj['signal'][read_obj['window'][0]:read_obj['window'][1]]>read_obj['OpenPore'])]
    obj = delete_reads_in_an_obj(obj=obj,  reads_need_to_remove=read_ids_with_signal_higher_than_i0)
    return obj


def find_sd_left_right_cutoff_for_an_att(
    obj: dict,
    att: str,
    sd_fold: float = 1.0,
):
    att_values = []
    for read_id, read_obj in obj.items():
        att_values.append(read_obj[att])
    att_values = np.array(att_values)
    mu, std = np.mean(att_values), np.std(att_values)
    left, right = mu-std*sd_fold, mu+std*sd_fold
    return left, right

def select_reads_within_att_range(
    obj: dict,
    att: str,
    left: float,
    right: float,
):
    reads_need_to_be_removed = []
    for read_id, read_obj in obj.items():
        if read_obj[att]<left or read_obj[att]>right:
            reads_need_to_be_removed.append(read_id)
    obj = delete_reads_in_an_obj(obj=obj, reads_need_to_remove=reads_need_to_be_removed)
    return obj


def get_signals_for_reads_in_an_obj(
    obj: dict, 
    att: str = 'signal',
    read_ids: list = [], 
    down_sample_to: int = 1000, 
    normalize_by_openpore: bool = True,
    target: Literal['dna1', 'window', 'dna2', 'all'] = 'window',
) -> list:
    """Extract signals (in windows, in dna1, in dna2 or all) of an obj. signal would be dowm sampled or normalized.

    Args:
        obj (dict): an obj
        att (str, optional): attribute name of the signals. Defaults to 'signal'.
        read_ids (list, optional): reads needed to be extracted signals in this obj. Defaults to [] meaning all reads.
        down_sample_to (int, optional): scale signal lenght to this value. Defaults to 1000.
        normalize_by_openpore (bool, optional): normalized or not. Defaults to True.
        target (Literal['dna1', 'window', 'dna2', 'all']): extract this part of the read. Defaults to 'window'.

    Returns:
        np.array: signals, 2d array. Each row represents a read. Each read contains a series of signals.
    """
    assert target in ['dna1', 'window', 'dna2', 'all']
    Xs = []
    if len(read_ids) == 0:
        read_ids = list(obj.keys())

    for read_id in read_ids:
        read_obj = obj[read_id]
        start, end = read_obj['window'][0], read_obj['window'][1]
        if target == 'window':
            target_start, target_end = start, end
        elif target == 'dna1':
            target_start, target_end = 0, start
        elif target == 'dna2':
            target_start, target_end = end, len(read_obj['signal'])
        elif target == 'all':
            target_start, target_end = 0, len(read_obj['signal'])
        X = down_sampling(read_obj[att][target_start:target_end], down_sample_to=down_sample_to)
        X[X<0] = 0

        if normalize_by_openpore:
            X /= read_obj['OpenPore']
            X[X>1] = 1
        Xs.append(list(X))
    Xs = np.array(Xs)
    return Xs



# ref to https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python
def _remove_spikes(
    signal: np.array,
    smooth_method: Literal['median_filter', 'ewma_fb'] = 'median_filter',
    high_clip: float = 0.8, 
    low_clip: float = 0.0,
    span: int = 100,
    delta: float = 0.05,
):
    clipped_signal = clip_data(signal, high_clip=high_clip, low_clip=low_clip)
    if smooth_method == 'median_filter':
        smooth_signal = median_filter(clipped_signal, size=span)
    elif smooth_method == 'ewma_fb':
        smooth_signal = ewma_fb(clipped_signal, span=span)
    # remove_outlier_signal = remove_outliers(clipped_signal, smooth_signal, delta=delta)
    # clean_signal = pd.Series(remove_outlier_signal).interpolate().values
    clean_signal = np.where(np.abs(clipped_signal - smooth_signal) > delta, smooth_signal, clipped_signal)
    return clean_signal, clipped_signal, smooth_signal

def remove_spikes(
    signal: np.array,
    smooth_method: Literal['median_filter', 'ewma_fb'] = 'median_filter',
    high_clip: float = 300.0, 
    low_clip: float = 0.0,
    span: int = 30,
    delta: float = 20,
    head_keeping_len: int = 200,
    tail_keeping_len: int = 200,
) -> np.array:
    """remove spikes within the signal. the head and tail of signal would not be changed.

    Args:
        signal (np.array): 1d array
        high_clip (float, optional): change values higher than this value as this value. Defaults to 300.0.
        low_clip (float, optional): change values lower than this value as this value. Defaults to 0.0.
        span (int, optional): the span of ewm or size of median_filter. Defaults to 100. suggest 100 for ewm and 30 for median_filter.
        delta (float, optional): if the distance between smoothed and raw value larger than this value,
                                set raw value as smoothed value. Defaults to 20.
        head_keeping_len (int, optional): the head of this number would not be changes. Defaults to 200.
        tail_keeping_len (int, optional): the tail of this number would not be changes. Defaults to 200.

    Returns:
        np.array: signal after removing spikes
    """
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    clean_signal, clipped_signal, smooth_signal = _remove_spikes(
        signal=signal,
        smooth_method=smooth_method,
        high_clip=high_clip, 
        low_clip=low_clip,
        span=span,
        delta=delta,
    )
    clean_signal[0:head_keeping_len] = signal[0:head_keeping_len]
    clean_signal[-tail_keeping_len:] = signal[-tail_keeping_len:]
    return clean_signal
    



def ewma_fb(x, span):
    ''' Apply forwards, backwards exponential weighted moving average (EWMA) to df_column. '''
    # Forwards EWMA.
    fwd = pd.Series(x).ewm(span=span).mean()
    # Backwards EWMA.
    bwd = pd.Series(x[::-1]).ewm(span=span).mean()
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack(( fwd, bwd[::-1] ))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma

# def remove_outliers(spikey, fbewma, delta):
#     ''' Remove data from df_spikey that is > delta from fbewma. '''
#     cond_delta = (np.abs(spikey-fbewma) > delta)
#     np_remove_outliers = np.where(cond_delta, np.nan, spikey)
#     return np_remove_outliers

def clip_data(unclipped, high_clip, low_clip):
    ''' Clip unclipped between high_clip and low_clip. 
    unclipped contains a single column of unclipped data.'''
    
    # clip data above HIGH_CLIP or below LOW_CLIP
    # cond_clip = (unclipped > high_clip) | (unclipped < low_clip)
    # np_clipped = np.where(cond_clip, np.nan, unclipped)
    np_clipped = np.clip(unclipped, low_clip, high_clip)
    return np_clipped


def extract_read_with_att_range(
    obj: dict,
    att: str,
    att_min: float = -np.inf,
    att_max: float = np.inf,
) -> dict:
    assert att in obj[list(obj.keys())[0]]

    new_obj = {}
    for read_id, read_obj in obj.items():
        if read_obj[att] >= att_min and read_obj[att] <= att_max:
            new_obj[read_id] = read_obj
    
    return new_obj



def smooth_signal_for_an_obj(
    obj: dict,
    in_place: bool = False,
    span: int = 200,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    
    for read_id, read_obj in new_obj.items():
        smoothed_signal = ewma_fb(read_obj['signal'], span=span)
        read_obj['smoothed_signal'] = np.array(pd.Series(smoothed_signal).interpolate())
    
    if in_place:
        return None
    else:
        return new_obj
    

def smooth_signal_by_median_filter_for_an_obj(
    obj: dict,
    in_place: bool = False,
    span: int = 100,
    att: str = 'signal',
    new_att: str = 'smoothed_signal',
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    
    for read_id, read_obj in new_obj.items():
        smoothed_signal = median_filter(read_obj[att].astype(np.float32), size=span)
        read_obj[new_att] = smoothed_signal
    
    if in_place:
        return None
    else:
        return new_obj

def get_I2Io_for_an_obj(
    obj: dict,
    att: str = 'signal',
    new_att: str = 'signal_I2Io',
    in_place: bool = False,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        x = read_obj[att]
        read_obj[new_att] = x / read_obj['OpenPore']

    if in_place:
        return None
    else:
        return new_obj


def down_sampling(
    array: np.array, 
    down_sample_to: int = 1000
) -> np.array:
    total_length = len(array)
    
    if total_length < down_sample_to:
        return np.concatenate((array,[0 for i in range(down_sample_to-total_length)]))
    
    sample_idx = np.round(np.linspace(start=0, stop=total_length-1, num=down_sample_to)).astype(np.int16)
    return array[sample_idx]


def down_sample_signal_for_an_obj(
    obj: dict,
    att: str = 'signal',
    new_att: str = 'dowm_sample',
    down_sample_to: int = 1000,
    target_window: bool = True,
    in_place: bool = False,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    
    for read_id, read_obj in new_obj.items():
        x = read_obj[att]
        if target_window:
            s, e = read_obj['window']
            x = x[s:e]
        x = down_sampling(x, down_sample_to=down_sample_to)
        read_obj[new_att] = x
    
    if in_place:
        return None
    else:
        return new_obj


def extract_x_reads_randomly(
    obj: dict,
    seed: int = 0,
    read_num: int = 100,
) -> dict:
    np.random.seed(seed)
    all_read_ids = list(obj.keys())
    np.random.shuffle(all_read_ids)
    select_read_ids = all_read_ids[0:read_num]
    sub_obj = extract_reads_as_an_obj(
        obj=obj,
        read_ids=select_read_ids,
    )
    return sub_obj


def get_heights_of_first_stairs_for_an_obj(
    obj: dict, 
    sample_name: str = 'sample'
) -> pd.DataFrame:
    read_ids, heights = [], []
    for read_id, read_obj in obj.items():
        stair_signal = read_obj['stair_signal'] / read_obj['OpenPore']
        heights.append(stair_signal[0])
        read_ids.append(read_id)
    h_df = pd.DataFrame({'read_id': read_ids, 'first_stair': heights})
    h_df['sample'] = sample_name
    return h_df


def nor_stair_signal_for_an_obj_by_first_stair(
    obj: dict,
    first_stair_height: float = 0.3,
    in_place: bool = False,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        read_obj['stair_signal_norm'] = read_obj['stair_signal'] / read_obj['stair_signal'][0] * first_stair_height

    if in_place:
        return None
    else:
        return new_obj
    

def get_nor_stair_height_df(
    obj: dict,
    target_signal_att: str = 'stair_signal_norm',
) -> pd.DataFrame:
    stair_height_dfs = []
    for read_id, read_obj in obj.items():
        stair_height_dfs.append(read_obj[target_signal_att][read_obj['transitions'][0:-1]])
    stair_height_dfs = pd.DataFrame(stair_height_dfs)
    stair_height_dfs.columns = [f'stair{i}' for i in range(stair_height_dfs.shape[1])]
    stair_height_dfs = stair_height_dfs.stack().reset_index()
    stair_height_dfs.columns = ['read_id', 'stair_num', 'stair_height']
    return stair_height_dfs



def filter_obj_by_stair_height_cutoffs(
    obj: dict,
    cutoff_config,
    target_pep: str,
) -> dict:
    new_obj = {}

    for read_id, read_obj in obj.items():
        stair_heights = read_obj['stair_signal_norm'][read_obj['transitions'][0:-1]]
        flag = True
        for i, stair_height in enumerate(stair_heights):
            if i == 0:
                continue
            stair_min = cutoff_config.getfloat(target_pep, f'stair{i}_min')
            stair_max = cutoff_config.getfloat(target_pep, f'stair{i}_max')
            if stair_height < stair_min or stair_height > stair_max:
                flag = False
                break
        if flag:
            new_obj[read_id] = read_obj

    return new_obj

        
    
def sample_1d_array(
    X: np.ndarray,
    sample_num: int = 1000,
) -> np.ndarray:
    x = np.linspace(0, 1, num=len(X))
    f = interpolate.interp1d(x, X)
    new_x = np.linspace(0, 1, num=sample_num)
    new_X = f(new_x)
    return new_X


def get_bound_by_iqr(
    a: np.array,
) -> Tuple[float, float]:
    q1 = np.quantile(a, 0.25)
    q3 = np.quantile(a, 0.75)
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    return lower_bound, upper_bound


def get_iqr(
    a: np.array
) -> float:
    q1 = np.quantile(a, 0.25)
    q3 = np.quantile(a, 0.75)
    iqr = q3-q1

    return iqr


def set_min_max_stair_signal_for_an_obj(
    obj: dict,
    target_att: str = 'stair_signal',
    new_att: str = 'min_max_stair_signal',
) -> None:
    for read_id, read_obj in obj.items():
        stair0_index, stair_second_to_last_index = read_obj['transitions'][0], read_obj['transitions'][-3]
        stair0_signal = read_obj['stair_signal'][stair0_index]
        stair_second_to_last_signal = read_obj['stair_signal'][stair_second_to_last_index]
        read_obj[new_att] = (read_obj[target_att] - stair0_signal) / (stair_second_to_last_signal - stair0_signal)
    return None


def remover_s0_stair_signal_for_an_obj(
    obj: dict,
    att: str,
    new_att: str,
) -> None:
    for read_id, read_obj in obj.items():
        read_obj[new_att] = read_obj[att][read_obj['transitions'][1]:]
    return None




def find_peak_with_scipy(
    x: np.ndarray,
    median_filter_size: int = 100,
    height: float = 100.0,
    prominence: float = 50,
    width: float = 30,
    distance: int = 100,
) -> np.ndarray:
    x_no_spike = remove_spikes(x)
    x_smooth = median_filter(x_no_spike, size=median_filter_size)
    peak_indexs, _ = find_peaks(x_smooth, height=height, 
                                prominence=prominence, width=width, 
                                distance=distance)
    return peak_indexs

def find_first_x_for_an_obj(
    obj: dict,
    median_filter_size: int = 100,
    height_of_io: float = 0.4698,
    prominence: float = 50,
    width: float = 30,
    distance: int = 10,
) -> None:
    for read_id, read_obj in obj.items():
        x, io = read_obj['signal'], read_obj['OpenPore']
        height = height_of_io * io
        peak_indexs = find_peak_with_scipy(x, median_filter_size, height, prominence, width, distance)
        if len(peak_indexs) > 0:
            read_obj['window'] = [peak_indexs[0], len(x)-50]
        else:
            read_obj['window'] = None


def otsu_thresholding(X):
    # 初始化类间方差和最佳阈值
    max_variance = 0
    best_threshold = 0

    for i in X:
        # 计算前景和背景的像素数
        w0 = np.sum(X<i)
        w1 = len(X) - w0

        # 如果前景或背景像素数为0，跳过
        if w0 == 0 or w1 == 0:
            continue

        # 计算前景和背景的平均灰度值
        m0 = np.mean(X[X<i])
        m1 = np.mean(X[X>=i])

        # 计算类间方差
        variance = w0 * w1 * (m0 - m1) ** 2

        # 如果当前方差大于最大方差，更新最大方差和最佳阈值
        if variance > max_variance:
            max_variance = variance
            best_threshold = i
    return best_threshold


def combine_objs(
    obj_list: List[Union[dict, str]], 
    sample_name_list: list = [],
    label_list: list = [],
    get_obj_stat: bool = True,
) -> dict:
    """Combine several objs into one by adding different name to reads in each obj

    Args:
        obj_list (List[dict, str]): a list of obj
        sample_name_list (list): a list of sample name, will be added as prefix of read ids with "_" as linking
        label_list: (list): a list of label which would be used as y in model input, such as pepetide name.
                            Would be add to 'label' key in reads.  

    Returns:
        dict: _description_
    """
    sample_name_list = copy.deepcopy(sample_name_list)
    label_list = copy.deepcopy(label_list)
    sample_name_list = np.array(sample_name_list)

    for i, one in enumerate(obj_list):
        if isinstance(one, str):
            obj_list[i] = io.read_pickle(one)

    if len(sample_name_list) > 0:
        assert len(sample_name_list) == len(obj_list)

    if label_list:
        assert len(label_list) == len(obj_list)
        pp.add_label(objs=obj_list, labels=label_list)

    for indx in range(len(obj_list)):
        if len(sample_name_list) > 0:
            obj_list[indx] = {sample_name_list[indx] + "_" + k: v for k, v in obj_list[indx].items()}

    com_obj = copy.deepcopy(obj_list[0])

    for indx in range(1, len(obj_list)):
        com_obj.update(obj_list[indx])

    if get_obj_stat:
        state_obj(obj=com_obj)

    return com_obj

def state_obj(
    obj,
    des: str = None,
    return_sample_num_for_classes: bool = False
):
    """Print out total read number of an obj and read numbers of all classes if 'label' in obj

    Args:
        obj (_type_): _description_
        des (str): _description_
    """
    if des:
        print(des + "(read number):", end='\t')

    total_reads_num = len(obj)
    if des:
        print(f'{total_reads_num}')
    else:
        print(f'read number: {total_reads_num}')

    if 'label' in list(obj.values())[0]:
        df = pd.DataFrame([[read_id, read_obj['label']] for read_id, read_obj in obj.items()], columns=['read_id', 'label'])
        print(df['label'].value_counts())
        if return_sample_num_for_classes:
            return df['label'].value_counts()




def butter_lowpass(cutoff, fs, order=2):
    #这里假设采样频率为fs=5000hz,要滤除1000hz以上频率成分，即截至频率为1000hz,则wn=2*1000/5000=0.4
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=1000, fs=5000, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpass_filter_for_an_obj(
    obj: dict,
    att: str = 'signal',
    new_att: str = 'lowpass_signal',
    cutoff: float = 100,
) -> None:
    for read_id, read_obj in obj.items():
        read_obj[new_att] = butter_lowpass_filter(read_obj[att], cutoff=cutoff).astype(np.float16)
    return None


def get_all_dirs_in_a_dir(
    dir: str,
) -> list:
    # 指定目录路径
    dir = Path(dir)

    # 获取目录下的所有文件夹，并获取它们的修改时间
    folders = [(entry.name, entry.stat().st_mtime)
            for entry in dir.iterdir() if entry.is_dir()]

    # 按修改时间排序（升序）
    folders.sort(key=lambda x: x[1])


    all_folders = [dir.joinpath(str(folder_name)).as_posix() for folder_name, timestamp in folders]
    return all_folders


def random_extract_reads(
    obj: dict, 
    extract_num: int, 
    # validated_read: bool = True, 
    seed: int = 1
) -> dict:
    """Extract reads randomly from an obj.

    Args:
        obj_dict (_type_): obj
        extract_num (int): the number of reads would be selected
        # validated_read (bool, optional): Whether to only select reads with windows. Defaults to True.
        seed (int, optional): random seed. Defaults to 1.

    Returns:
        dict: _description_
    """
    # obj = copy.deepcopy(obj)

    # if validated_read:
    #     pp.filter_out_reads_without_widows(obj)

    read_ids = [read_id for read_id, read_obj in obj.items()]
    random.seed(seed)
    random.shuffle(read_ids)
    
    selected_dict = {}
    for read_id in read_ids[0:extract_num]:
        selected_dict[read_id] = obj[read_id]
    return selected_dict



def get_if_has_9x(
    obj: dict,
    min_9x_len: int = 50,
):
    df = []
    for read_id, read_obj in obj.items():
        signal = read_obj['signal'] / read_obj['OpenPore']
        s, e = read_obj['window']
        signal = signal[s:e]
        cutoff = np.mean((signal[0], signal[-1]))
        high_len = np.sum(signal > cutoff)
        has_9x = 0
        if high_len >= 50:
            has_9x = 1
        df.append([read_id, has_9x])
    df = pd.DataFrame(df, columns=['read_id', 'has_9x'])
    return df