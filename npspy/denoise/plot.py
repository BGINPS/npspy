#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: plot.py
@Description: description of this file
@Datatime: 2025/06/16 16:00:11
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"

from ..plot import draw_one_read, create_dir_if_not_exist, draw_window, draw_signal
from . import tools as tl

def draw_one_read_with_window_and_diff(
    obj: dict,
    read_id: str,
    figsize: Tuple[float, float] = (15,8),
    save_figure: bool = False, 
    save_dir: str = "./",
    save_file_name: str = 'signal.pdf',
    title = None, 
    ax = None,
    window_color: str = 'red',
    scale_by_openpore: bool = False,
    att_to_draw: str = 'signal',
    only_window: bool = False 
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    x = obj[read_id][att_to_draw]
    if scale_by_openpore:
        x = x / obj[read_id]['OpenPore']
    if only_window:
        s, e = obj[read_id]['window']
        x = x[s:e]
    draw_signal(x, ax=ax, color='#99AB5F', ylabel='I/I0')

    # draw window
    if not only_window:
        if 'window' in obj[read_id].keys():
            draw_window(obj, read_id, ax=ax, color=window_color, scale_by_openpore=scale_by_openpore)

    # draw diff
    if only_window:
        diff_signal = tl.get_diff_for_a_read_obj(obj[read_id], only_window=True)
    else:
        diff_signal = tl.get_diff_for_a_read_obj(obj[read_id], only_window=False)
    ax.plot(diff_signal, color='blue', label='diff_signal')

    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_file_name}')
        plt.close()
    else:
        return ax