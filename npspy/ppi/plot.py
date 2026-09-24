#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: plot.py
@Description: description of this file
@Datatime: 2025/06/23 11:47:12
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
from matplotlib.gridspec import GridSpec

def draw_scatterplot(
    low_high_df: pd.DataFrame,
):
    x = low_high_df['low_mean']
    y = low_high_df['high_mean']
    # 创建图形和网格布局
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4)
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_histx = fig.add_subplot(gs[0, 0:3])
    ax_histy = fig.add_subplot(gs[1:4, 3])

    # 主散点图
    ax_scatter.scatter(x, y, alpha=1, s=4)
    ax_scatter.set_xlabel('low_mean')
    ax_scatter.set_ylabel('high_mean')

    # X轴投影(顶部直方图)
    ax_histx.hist(x, bins=50, density=True, color='skyblue', edgecolor='black')
    ax_histx.set_xticks([])
    # ax_histx.set_yticks([])

    # Y轴投影(右侧直方图)
    ax_histy.hist(y, bins=50, density=True, orientation='horizontal', 
                color='salmon', edgecolor='black')
    # ax_histy.set_xticks([])
    ax_histy.set_yticks([])

    # 调整间距
    plt.tight_layout()