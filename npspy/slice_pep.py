import pickle
import numpy as np
import pandas as pd
import copy
import os
import sys
import re
import glob
import configparser
from configparser import ConfigParser, ExtendedInterpolation
from typing import Dict, Optional, Tuple, Union, Literal, List




def get_pep_end_by_diff(
    x: np.array,
    min_end_proportion: float = 0.35,
    max_end_proportion: float = 0.85,
) -> int:
    """get end point of a signal by the first diff

    Args:
        x (np.array): signal
        min_end_proportion (float, optional): end >= len(x)*this_value. Defaults to 0.35.
        max_end_proportion (float, optional): end <= len(x)*this_value. Defaults to 0.85.

    Returns:
        int: _description_
    """

    try:
        import step_wave_tools as sw
    except ModuleNotFoundError:
        print('step_wave_tools has not been installed. Please install it first according to https://gitlab.genomics.cn/panhailin/step_wave_tools')
        sys.exit(1)
    # sys.path.insert(0, '/Users/hailinpan/Documents/git_hub/step_wave_tools')
    # import step_wave_tools as sw

    dtw_x = sw.find_window.DWT_transform(x)
    staircasing, _, step_width = sw.find_window.terracing(dtw_x)
    first_diff = np.diff(staircasing)
    min_diff_idx_end = np.argmin(first_diff[int(min_end_proportion * (len(first_diff))): int(max_end_proportion * (len(first_diff)))]) + int(min_end_proportion * len(first_diff))
    end = min_diff_idx_end
    for i in range(min_diff_idx_end, min_diff_idx_end-500, -1):
        if x[i] >= x[end]:
            end = i
        else:
            break
    return end

def get_pep_end_to_window_end_median_for_an_obj(
    obj: dict,
    min_pep_end_proportion_in_window: float = 0.35,
    max_pep_end_proportion_in_window: float = 0.85,
) -> np.array:
    all_medians = []
    for read_id, read_obj in obj.items():
        x = read_obj['signal']/read_obj['OpenPore']
        s, e = read_obj['window']
        x = x[s:e]
        end = get_pep_end_by_diff(x, min_end_proportion=min_pep_end_proportion_in_window,
                                  max_end_proportion=max_pep_end_proportion_in_window)
        all_medians.append(np.median(x[end:]))
    all_medians = np.array(all_medians)
    return all_medians


def get_pep_start(
    x: np.array,
    end: int,
    min_start_proportion: float = 0.2,
    max_start_proportion: float = 0.5,
    min_start_median: float = 0.26,
    max_start_median: float = 0.34,
) -> Union[int, None]:
    start = None

    start_start = int(len(x) * min_start_proportion)
    start_end = int(len(x) * max_start_proportion)

    for i in range(start_start, start_end):
        median_of_i = np.median(x[i:i+200])
        if median_of_i >= min_start_median and median_of_i <= max_start_median:
            start = i + 200
            break
    if start != None and start >= end:
        start = None
    return start

def get_pep_start_end_for_an_obj(
    obj: dict,
    min_pep_end_proportion_in_window: float = 0.35,
    max_pep_end_proportion_in_window: float = 0.85,
    min_pep_start_proportion_in_window: float = 0.2,
    max_pep_start_proportion_in_window: float = 0.5,
    min_start_median: float = 0.26,
    max_start_median: float = 0.34,
):
    all_read_ids = list(obj.keys())
    for read_id in all_read_ids:
        read_obj = obj[read_id]
        x = read_obj['signal']/read_obj['OpenPore']
        s, e = read_obj['window']
        x = x[s:e]
        end = get_pep_end_by_diff(x=x, 
                                  min_end_proportion=min_pep_end_proportion_in_window,
                                  max_end_proportion=max_pep_end_proportion_in_window)
        start = get_pep_start(x=x,
                              end=end,
                              min_start_proportion=min_pep_start_proportion_in_window,
                              max_start_proportion=max_pep_start_proportion_in_window,
                              min_start_median=min_start_median,
                              max_start_median=max_start_median)
        if start == None:
            del obj[read_id]
        else:
            read_obj['window'] = [start, end]
            read_obj['signal'] = x * read_obj['OpenPore']
            for one_att in ['mean_of_I/I0', 'median_of_I/I0', 'std_of_I/I0', 'window_length']:
                if one_att in read_obj:
                    del read_obj[one_att]

        
        

        


