#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: plot.py
@Description: description of this file
@Datatime: 2025/07/03 15:32:32
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"    
import matplotlib as mpl


from .. import plot as pl

def draw_reconstruction_error_distribution(
    error_df: pd.DataFrame,
    threshold: float,
    save_dir: str = '.',
    save_name: str = 'reconstruction_error_distribution.pdf'
):
    """
    Draw the distribution of reconstruction error.
    
    :param error_df: DataFrame containing reconstruction errors.
    :param save_dir: Directory to save the plot.
    :param save_name: Name of the saved plot file.
    """
    plt.figure(figsize=(5, 4))
    sns.kdeplot(error_df['reconstruction_error'])
    plt.axvline(threshold, color='red', linestyle='--')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Density')
    pl.create_dir_if_not_exist(save_dir)
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')
    plt.close()


def draw_mahalanobis_distance_distribution(
    mahalanobis_dist_df: pd.DataFrame,
    threshold: float,
    save_dir: str = '.',
    save_name: str = 'mahalanobis_distance_distribution.pdf',
):

    plt.figure(figsize=(5, 4))
    sns.kdeplot(mahalanobis_dist_df['mahalanobis_distance'])
    plt.axvline(threshold, color='red', linestyle='--')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('Density')
    pl.create_dir_if_not_exist(save_dir)
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')
    plt.close()


def draw_lzc_distribution(
    lzc_df: pd.DataFrame,
    threshold: float,
    save_dir: str = '.',
    save_name: str = 'lzc_distribution.pdf',
):

    plt.figure(figsize=(5, 4))
    sns.kdeplot(lzc_df['lzc'])
    plt.axvline(threshold, color='red', linestyle='--')
    plt.xlabel('Lempel-Ziv complexity')
    plt.ylabel('Density')
    pl.create_dir_if_not_exist(save_dir)
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')
    plt.close()


def draw_radar_plot(
    metrics_df: pd.DataFrame,
    figsize: Tuple[float, float] = (8.0, 8.0),
    y_min: float = 0.0,
    y_max: float = 1.0,
    colors: List[str] = None,
    linestyles: Optional[List[str]] = None,
    marker: Optional[List[str]] = 'o',
    legned_ncol: int = 2,
    y_tick_num: int = 5,
    alpha: float = 1.0,
):
    labels = metrics_df.columns.tolist()

    # 2. 计算顶点角度（五边形）
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 3. 数据处理（闭合)
    datas = []
    for i in range(len(metrics_df)):
        values = metrics_df.iloc[i].values.flatten().tolist()
        values += values[:1]
        datas.append(values)
    
    # 4. 创建极坐标图
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # 5. 绘制五边形框架
    for one_x in np.arange(y_min, y_max + 1e-5, (y_max-y_min)/y_tick_num):
        ax.plot(angles, [one_x]*len(angles), color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylim(y_min, y_max)  # 设置极坐标的半径范围
    # ax.set_yticks(np.arange(y_min, y_max + 1e-5, (y_max-y_min)/5), [round(i, 2) for i in np.arange(y_min, y_max + 1e-5, (y_max-y_min)/5)])
    ax.set_yticklabels([])  # 隐藏半径标签

    # 6. 绘制数据区域

    if colors is None:
        if linestyles is None:
            linestyles = ['solid'] * len(metrics_df)
        for data_label, data, linestyle in zip(metrics_df.index, datas, linestyles):
            ax.plot(angles, data, marker=marker, linewidth=1, label=data_label, linestyle=linestyle, alpha=alpha)
    else:
        if linestyles is None:
            linestyles = ['solid'] * len(metrics_df)
        for data_label, data, c, linestyle in zip(metrics_df.index, datas, colors, linestyles):
            ax.plot(angles, data, marker=marker, linewidth=1, label=data_label, color=c, linestyle=linestyle)

    for value in [round(i, 2) for i in np.arange(y_min, y_max + 1e-5, (y_max-y_min)/y_tick_num)]:
        # 在0°方向添加刻度标签
        ax.text(np.radians(0), value, f' {value}',
                verticalalignment='center', 
                horizontalalignment='left', 
                zorder=10)    
    
    # 7. 设置五边形顶点标签
    ax.set_xticks(angles[:-1])  # 去掉最后一个重复点
    ax.set_xticklabels(labels)

    # 8. 隐藏圆形网格线，显示五边形
    ax.yaxis.grid(False)  # 隐藏圆形网格
    ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location('N')  # 0度位置在顶部
    ax.set_rlabel_position(0)        # 半径标签位置

    plt.legend(ncol=legned_ncol, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    # ax.legend(handles=line_handles, ncol=legned_ncol, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()

    return ax


def draw_radar_plot_with_two_cmps(
    metrics_df: pd.DataFrame,
    figsize: Tuple[float, float] = (8.0, 8.0),
    y_min: float = 0.0,
    y_max: float = 1.0,
    cmps: List[str] = ['Reds', 'Blues'],
):
    n_for_each_class = len(metrics_df) // 2
    cmap = mpl.colormaps[cmps[0]]
    colors1 = cmap(np.arange(cmap.N, 100, -(cmap.N-100)//n_for_each_class)) 

    cmap = mpl.colormaps[cmps[1]]
    colors2 = cmap(np.arange(cmap.N, 100, -(cmap.N-100)//n_for_each_class))

    all_colors = []
    all_colors += colors1.tolist()
    all_colors += colors2.tolist()
    return draw_radar_plot(
        metrics_df=metrics_df,
        figsize=figsize,
        y_min=y_min,
        y_max=y_max,
        colors=all_colors
    )