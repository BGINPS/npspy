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
import h5py
from multiprocessing import Pool
from functools import partial


from . import io
from . import stat



def _get_reads_dict_from_fast5(
    fast5_path: str, 
    retry: int =2
):
    while retry > 0:
        try:
            reads_dict = dict()
            with h5py.File(fast5_path, 'r') as f:
                if fast5_path.endswith('fast5'):
                    for read_id, val in f['Raw']['Reads'].items():
                        if val is not None:
                            fast5_fn = os.path.basename(fast5_path).replace('.fast5', '')
                            ch_read_id = f'{fast5_fn}_Read_{read_id[5:]}' if fast5_fn.startswith('channel') else read_id
                            signal = np.array(f['Raw']['Reads'][read_id]['Signal'])
                            openpore = np.array(f['Raw']['Reads'][read_id]['Openpore_mean']).item()
                            reads_dict[ch_read_id] = {'window': None, 'confidence': 0, 'signal': signal,
                                                      'OpenPore': openpore}
                else:
                    raise Exception(f'Invalid file extension: {fast5_path}')
                f.close()
            return reads_dict
        except Exception as e:
            retry -= 1
            print(f'[Warning] Failed reading reads file: {fast5_path}. Retry({retry})')
    return None


def get_reads_dict_from_fast5_dir(
    fast5_dir: str, # ...results/Meta/
    pool_num: int = 3,
):
    fast5_paths = glob.glob(f'{fast5_dir}/*.fast5')
    with Pool(min(len(fast5_paths), pool_num)) as p:
        reads_dict_list = p.starmap(_get_reads_dict_from_fast5, zip(fast5_paths))
    result = dict()
    for d in reads_dict_list:
        result.update(d)
    return result



def do_slice(signal, openpore=220.0, offset=0.4698, start_trim_ratio=0.3, end_trim_ratio=0.075,
             find_peak_min_peak_len: int = 15,
             find_peak_min_lagging_len: int = 200,

):
    y_cut = openpore * offset
    start_trim_len = round(start_trim_ratio * len(signal))
    end_trim_len = round(end_trim_ratio * len(signal))
    signal_trimmed = signal[start_trim_len:-end_trim_len]

    left = find_peak(signal_trimmed, y_cut, min_peak_len=find_peak_min_peak_len, min_lagging_len=find_peak_min_lagging_len)
    if left is None:
        return None
    else:
        left += start_trim_len

    right = find_peak(signal_trimmed[::-1], y_cut, min_peak_len=find_peak_min_peak_len, min_lagging_len=find_peak_min_lagging_len)
    if right is None:
        return None
    else:
        right = len(signal) - end_trim_len - right

    if left + 600 >= right:
        return None
    else:
        return left, right
    




def _find_platforms(
    x: np.array,
    min_c: float = 0.4698,
    max_c: float = 1.0,
    min_len: int = 15,
    min_distance: int = 200
) -> Union[np.array, None]: #[[start, end],[start, end]]   [)
    """find platform for a signal.
    1. all signals in a platform should be in the range of [min_c, max_c]
    2. the length of a platform should be >= min_len
    3. in a case platform A is followed by platform B
       if the start of B - the end of A should >= min_distance:
            A would be in the result
        else:
            remove A

    Args:
        x (np.array): signal
        min_c (float, optional): the min cutoff for platform. Defaults to 0.4698.
        max_c (float, optional): the max cutoff for platform. Defaults to 1.0.
        min_len (int, optional): the min len for platform. Defaults to 15.
        min_distance (int, optional): see  point `3`. Defaults to 200.

    Returns:
        Union[np.array, None]: [[s,e],[s,e]] or None
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
        if np.all(one_seg>min_c) and np.all(one_seg<=max_c) and len(one_seg)>=min_len:
            filtered_segments.append([segments[i],segments[i+1]])
    
    if len(filtered_segments) == 0:
        return None

    # filter according distance to next segment
    filter_dis_segments = []
    for i in range(len(filtered_segments)-1):
        if filtered_segments[i+1][0] - filtered_segments[i][1] >= min_distance:
            filter_dis_segments.append(filtered_segments[i])
    filter_dis_segments.append(filtered_segments[-1])

    filter_dis_segments = np.array(filter_dis_segments, dtype=np.int32)

    return filter_dis_segments

def find_peak(signal, y_cut, max_c: float=250, min_peak_len = 15, min_lagging_len = 200):
    platform_segments = _find_platforms(x=signal, min_c=y_cut, max_c=max_c, min_len=min_peak_len,
                                        min_distance=min_lagging_len)
    if isinstance(platform_segments, (np.ndarray,list)):
        return platform_segments[0][1]
    return None


def slice_peptide(
    obj: dict, 
    process_all_reads: bool = True, 
    pool_num: int = 3,
    offset: float = 0.4698, 
    start_trim_ratio: float = 0.3, 
    end_trim_ratio: float = 0.075,
    find_peak_min_peak_len: int = 15,
    find_peak_min_lagging_len: int = 200,
):
    keys = obj.keys()
    signal_openpore = [(v['signal'], v['OpenPore']) for v in obj.values()]
    do_slice_with_para = partial(do_slice, offset=offset, start_trim_ratio=start_trim_ratio,
                                 end_trim_ratio=end_trim_ratio,
                                 find_peak_min_peak_len=find_peak_min_peak_len,
                                 find_peak_min_lagging_len=find_peak_min_lagging_len)
    with Pool(pool_num) as p:
        windows = p.starmap(do_slice_with_para, signal_openpore)
    sliced_count = 0
    for key, window in zip(keys, windows):
        if window is None:
            if not process_all_reads:
                del obj[key]
        else:
            sliced_count += 1
            obj[key]['window'] = window
    # return sliced_count


def get_obj_from_fast5_dir_and_find_window(
    fast5_dir: str,
    pool_num: int = 3,
    save: bool = False,
    save_dir: str = './',
    save_file_prefix: str = 'sample',
    offset: float = 0.4698, 
    start_trim_ratio: float = 0.3, 
    end_trim_ratio: float = 0.075,
    find_peak_min_peak_len: int = 15,
    find_peak_min_lagging_len: int = 200,
) -> dict:
    """read all fast5 files in a dir and return obj with windows

    Args:
        fast5_dir (str): ...results/Meta/
        pool_num (int, optional): cpu number. Defaults to 3.
        save (bool, optional): if to save obj as pkl file. Defaults to False.
        save_dir (str, optional): pkl dir to be saved. Defaults to './'.
        save_file_prefix (str, optional): pkl file prefix. Defaults to 'sample'.

    Returns:
        dict: obj
    """
    obj = get_reads_dict_from_fast5_dir(fast5_dir=fast5_dir, pool_num=pool_num)
    slice_peptide(obj=obj, pool_num=pool_num,
                  offset=offset,
                  start_trim_ratio=start_trim_ratio,
                  end_trim_ratio=end_trim_ratio,
                  find_peak_min_peak_len=find_peak_min_peak_len,
                  find_peak_min_lagging_len=find_peak_min_lagging_len,
                  )
    if save:
        io.save_pickle(obj, f'{save_dir}/{save_file_prefix}.pkl')
    return obj



def get_obj_from_fast5_dir_and_find_window_and_save_pkl_and_stat(
    fast5_dir: str,
    offset: float = 0.4698,
    pool_num: int = 1,
    save_dir: str = './',
    save_file_prefix: str = 'sample',
) -> None:
    obj = get_obj_from_fast5_dir_and_find_window(
        fast5_dir=fast5_dir,
        pool_num=pool_num,
        offset=offset,
        save = False,
        save_dir=save_dir,
        save_file_prefix=save_file_prefix,
    )

    stat.get_read_number(
        obj, return_df_or_dict='df', sample_name=save_file_prefix,
        save=True, save_dir=save_dir,
        save_file_name=f'{save_file_prefix}.pkl.stat.csv'
    )

    io.save_pickle(obj, f'{save_dir}/{save_file_prefix}.pkl')

    return None

    
