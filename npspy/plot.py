#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: plot.py
@Description: description of this file
@Datatime: 2024/12/17 10:19:42
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import os
import re
import gc
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as grid_spec
from . import tools as tl

def draw_signal(
    signal: np.array, 
    ax = None, 
    color: str = '#99AB5F',
    alpha: float = 1.0,
    ylabel: str = None,
):
    if ax:
        sns.lineplot(x=range(len(signal)),y=signal,ax=ax, color=color, alpha=alpha)
    else:
        ax = sns.lineplot(x=range(len(signal)),y=signal, color=color, alpha=alpha)
    if ylabel == None:
        ylabel = 'current'
    ax.set_xlabel('time (1/5000 s)')
    ax.set_ylabel(ylabel)

def draw_reset_stair_signal(
    signal: np.array, 
    ax = None, 
    color: str = '#99AB5F',
    alpha: float = 1.0,
    ylabel: str = None,
):
    signal = np.concatenate([signal, [signal[-1]]])
    
    if ax:
        sns.lineplot(x=range(len(signal)),y=signal,ax=ax, color=color, alpha=alpha, drawstyle='steps-post')
    else:
        ax = sns.lineplot(x=range(len(signal)),y=signal, color=color, alpha=alpha, drawstyle='steps-post')
    if ylabel == None:
        ylabel = 'current'
    ax.set_xlabel('stairs')
    ax.set_ylabel(ylabel)

def draw_reset_stair_signal_for_one_read(
    obj: dict,
    read_id: str,
    att_to_draw: str = 'reset_stair_signal',
    figsize: Tuple[float, float] = (15,8),
    color: str = '#99AB5F',
    save_figure: bool = False, 
    save_dir: str = "./",
    save_file_name: str = 'signal.pdf',
    title = None, 
    ax = None,
    scale_by_openpore: bool = False,
    **kwargs,
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    
    x = obj[read_id][att_to_draw]

    if scale_by_openpore:
        x = x / obj[read_id]['OpenPore']

    ylabel = 'I/I0' if scale_by_openpore else None
    draw_reset_stair_signal(x, ax=ax, color=color, ylabel=ylabel, **kwargs)
    
    ax.set_title(title)

    if save_figure:
        plt.savefig(f'{save_dir}/{save_file_name}')
    else:
        return ax
    

def draw_one_read(
    obj: dict,
    read_id: str,
    att_to_draw: str = 'signal',
    figsize: Tuple[float, float] = (15,8),
    color: str = '#99AB5F',
    save_figure: bool = False, 
    save_dir: str = "./",
    save_file_name: str = 'signal.pdf',
    title = None, 
    ax = None,
    scale_by_openpore: bool = False,
    **kwargs,
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    
    x = obj[read_id][att_to_draw]

    if scale_by_openpore:
        x = x / obj[read_id]['OpenPore']

    ylabel = 'I/I0' if scale_by_openpore else None
    draw_signal(x, ax=ax, color=color, ylabel=ylabel, **kwargs)
    
    ax.set_title(title)

    if save_figure:
        plt.savefig(f'{save_dir}/{save_file_name}')
    else:
        return ax

def draw_one_read_with_leading_platform_and_max_point(
    obj: dict,
    read_id: str,
    figsize: Tuple[float, float] = (15,8),
    save_figure: bool = False, 
    save_dir: str = "./",
    save_file_name: str = 'signal.pdf',
    title = None, 
    ax = None,
):
    ax = draw_one_read(obj=obj, read_id=read_id, figsize=figsize, save_figure=False,
                  save_dir=save_dir, save_file_name=save_file_name, title=title, ax=ax,)
    ax.hlines(y=obj[read_id]['OpenPore'], xmin=0, xmax=obj[read_id]['leading_platform_index'], color='blue')

    ax.plot(obj[read_id]['signal'].argmax(), obj[read_id]['signal'].max(), 'x', color='red')

    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_file_name}')
        plt.close()
    else:
        return ax

def draw_one_read_with_stair_signal(
    obj: dict,
    read_id: str,
    figsize: Tuple[float, float] = (15,8),
    save_figure: bool = False, 
    save_dir: str = "./",
    save_file_name: str = 'signal.pdf',
    title = None, 
    ax = None,
    scale_by_openpore: bool = True,
):
    ax = draw_one_read(obj=obj, read_id=read_id, figsize=figsize, save_figure=False,
                  save_dir=save_dir, save_file_name=save_file_name, title=title, ax=ax,
                  scale_by_openpore=scale_by_openpore)
    draw_one_read(obj, read_id, att_to_draw='stair_signal', ax=ax, color='red',
                  scale_by_openpore=scale_by_openpore)

    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_file_name}')
        plt.close('all')
        # gc.collect()
        # MatplotlibClearMemory()
    else:
        return ax


# def MatplotlibClearMemory():
#     #usedbackend = matplotlib.get_backend()
#     #matplotlib.use('Cairo')
#     allfignums = plt.get_fignums()
#     for i in allfignums:
#         fig = plt.figure(i)
#         fig.clear()
#         plt.close(fig)

    

def create_dir_if_not_exist(
    dir: str,
):
    if not os.path.exists(dir):
        os.makedirs(dir)


def draw_one_read_with_window(
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
    draw_one_read_kwargs: Union[dict, None] = {}
):
    ax = draw_one_read(obj=obj, read_id=read_id, figsize=figsize, save_figure=False,
                  save_dir=save_dir, save_file_name=save_file_name, title=title, ax=ax,
                  scale_by_openpore=scale_by_openpore, att_to_draw=att_to_draw,
                  **draw_one_read_kwargs)
    # read_obj = obj[read_id]

    # if read_obj['window'] is not None:
    #     start, end = read_obj['window']
        # ax.plot([start, end], [read_obj['signal'][start], read_obj['signal'][end]], 'x', color='red')

    # draw window
    if 'window' in obj[read_id].keys():
        draw_window(obj, read_id, ax=ax, color=window_color, scale_by_openpore=scale_by_openpore)

    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_file_name}')
        plt.close()
    else:
        return ax

def draw_window(
    obj: dict,
    read_id: str,
    ax,
    color: str = 'red',
    att_of_y_for_window: str = 'signal',
    scale_by_openpore: bool = False,
):
    read_obj = obj[read_id]
    if isinstance(read_obj['window'], (list, tuple)):
        window_start, window_end = read_obj['window']
        # ax.axvline(window_start, color='red', ls='--')
        # ax.axvline(window_end, color='red', ls='--')
        xs = [window_start, window_end]
        ys = [read_obj[att_of_y_for_window][window_start], read_obj[att_of_y_for_window][window_end]]
        if scale_by_openpore:
            ys = np.array(ys)/read_obj['OpenPore']
        ax.plot(xs, ys, 'x',
                color=color)

def draw_all_reads_in_an_obj(
    obj: dict,
    figsize: Tuple[float, float] = (15,8),
    save_dir: str = "./",
    save_file_name_postfix: str = '.signal.pdf',
    scale_by_openpore: bool = False,
    att_to_draw: str = 'signal',
):
    for read_id, read_obj in obj.items():
        draw_one_read_with_window(obj=obj, 
                                  read_id=read_id,
                                  figsize=figsize,
                                  save_figure=True,
                                  save_dir=save_dir,
                                  save_file_name=f'{read_id}{save_file_name_postfix}',
                                  scale_by_openpore=scale_by_openpore,
                                  att_to_draw=att_to_draw,
                                  )

def draw_random_x_reads_in_an_obj(
    obj: dict,
    seed: int = 0,
    random_read_num: int = 100,
    figsize: Tuple[float, float] = (15,8),
    save_dir: str = "./",
    save_file_name_postfix: str = '.signal.pdf',
    scale_by_openpore: bool = True,
    att_to_draw: str = 'signal',
):
    np.random.seed(seed)
    all_read_ids = list(obj.keys())
    np.random.shuffle(all_read_ids)
    select_read_ids = all_read_ids[0:random_read_num]
    sub_obj = tl.extract_reads_as_an_obj(
        obj=obj,
        read_ids=select_read_ids,
    )
    # if scale_by_openpore:
    #     for read_id, read_obj in sub_obj.items():
    #         read_obj['signal'] = read_obj['signal']/read_obj['OpenPore']

    draw_all_reads_in_an_obj(
        obj=sub_obj,
        figsize=figsize,
        save_dir=save_dir,
        save_file_name_postfix=save_file_name_postfix,
        scale_by_openpore=scale_by_openpore,
        att_to_draw=att_to_draw,
    )

def draw_random_x_reads_in_an_obj_with_specific_att_range(
    obj: dict,
    seed: int = 0,
    random_read_num: int = 100,
    att: str = 'mean_of_I2I0',
    att_min: float = 0.0,
    att_max: float = 1.0,
    figsize: Tuple[float, float] = (15,8),
    save_dir: str = "./",
    save_file_name_postfix: str = '.signal.pdf',
    scale_by_openpore: bool = True,
):
    sub_obj = tl.extract_read_with_att_range(
        obj=obj,
        att=att,
        att_min=att_min,
        att_max=att_max,
    )
    draw_random_x_reads_in_an_obj(
        obj=sub_obj,
        seed=seed,
        random_read_num=random_read_num,
        figsize=figsize,
        save_dir=save_dir,
        save_file_name_postfix=save_file_name_postfix,
        scale_by_openpore=scale_by_openpore
    )
    
        
def draw_all_reads_with_stair_signal_in_an_obj(
    obj: dict,
    figsize: Tuple[float, float] = (15,8),
    scale_by_openpore: bool = True,
    save_dir: str = "./",
    save_file_name_postfix: str = '.png',  
):
    for read_id, read_obj in obj.items():
        draw_one_read_with_stair_signal(obj=obj, 
                                        read_id=read_id,
                                        figsize=figsize,
                                        save_figure=True,
                                        save_dir=save_dir,
                                        save_file_name=f'{read_id}{save_file_name_postfix}',
                                        scale_by_openpore=scale_by_openpore,
                                  )
        

def draw_stair_num_histplot_for_an_obj(
    obj: dict,
    figsize: Tuple[float, float] = (5,5),
    save_figure: bool = False, 
    save_dir: str = "./",
    save_file_name_prefix: str = 'stair',
    ax = None,
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    
    x = [len(read_obj['transitions']) - 1 for read_id, read_obj in obj.items()]
    sns.histplot(data=x, kde=False, discrete=True, ax=ax)
    ax.text(0.7, 0.95, f'median: {np.median(x):.1f}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    ax.text(0.7, 0.90, f'mean: {np.mean(x):.1f}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    ax.text(0.7, 0.85, f'sd: {np.std(x):.1f}', ha='left', va='top', transform=ax.transAxes, fontsize=12)


    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_file_name_prefix}.stair_num.histplot.pdf')
        plt.close()
    else:
        return ax
    


def draw_ridge_plots(
    df: pd.DataFrame,
    class_att: str,
    x_att: str,
    targets: list,
    colors: list,
    figsize: Tuple[float, float] = (8,8),
    xlim: Tuple[float, float] = (0,1),
    ylim: Tuple[float, float] = (0,10),
    bandwidth: float = 0.05,
):
    gs = grid_spec.GridSpec(len(targets),1)
    fig = plt.figure(figsize=figsize)

    i = 0

    ax_objs = []
    for label in targets:
        # label = targets[i]
        x = np.array(df[df[class_att] == label][x_att])
        x_d = np.linspace(xlim[0], 1, 1000)

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(x[:, None])

        logprob = kde.score_samples(x_d[:, None])

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
        ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1, color=colors[i])


        # setting uniform x and y lims
        ax_objs[-1].set_xlim(xlim)
        ax_objs[-1].set_ylim(ylim)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_yticks([])
        

        if i == len(targets)-1:
            ax_objs[-1].set_xlabel(x_att,)# fontsize=16,)
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].set_xticks([])
            

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(xlim[0]-0.02,0, label, ha="right")


        i += 1

    gs.update(hspace=-0.5)


    # plt.tight_layout()
    # plt.show()



def draw_density_heatmap(
    data: np.ndarray,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (5,4), 
    save_figure: bool = False, 
    save_dir: str = "./",
    save_figure_name = None,
    cmap: str = 'YlGnBu',
    **kwargs,
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data, cmap=cmap, yticklabels=False, ax=ax, **kwargs)
    # ticks = [i for i in range(0,len(data[0])+1, int(len(data[0])/5))]
    # x_labels = [str(i) for i in ticks]
    # ax.set_xticks(ticks=ticks, labels=x_labels)
    # ax.set_yticks(ticks=ticks, labels=['1', '0.8', '0.6', '0.4', '0.2', '0'])
    ticks = [i for i in range(0,len(data[0])+1, int(len(data[0])/2))]
    x_labels = [str(i) for i in ticks]
    ax.set_xticks(ticks=ticks, labels=x_labels)
    ax.set_yticks(ticks=ticks, labels=['1', '0.5', '0'])
    ax.set_xlabel('scaled time')
    ax.set_ylabel('I/Io')

    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_figure_name}')


def draw_stack_plot(
    obj, 
    read_num: int = 3000, 
    color: str = 'black', 
    ax = None, 
    alpha: float = 0.007,
    figsize: Tuple[float, float] = (8,6), 
    save_figure: bool = False, 
    save_dir: str = "./",
    save_figure_name = None,
):
    obj = tl.extract_x_reads_randomly(obj, read_num=read_num)
    all_read_ids = list(obj.keys())
    window_sigs = tl.get_signals_for_reads_in_an_obj(obj=obj, read_ids=all_read_ids)
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim((0, len(window_sigs[0])))
    ax.set_ylim(0,1)
    ax.set_ylabel('I/I0')
    ax.set_xlabel('scaled time')
    for i, sig in enumerate(window_sigs):
        ax.plot(sig, color=color, linewidth=1, alpha=alpha)

    if save_figure:
        create_dir_if_not_exist(save_dir)
        plt.savefig(f'{save_dir}/{save_figure_name}')
        return None

    return ax


def line_plot_X_and_barycenter(X, barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)
    plt.ylim(0,1)


