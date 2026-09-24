from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def time_warp_augmentation(
    signal: np.ndarray, # 1d time series
    num_knots: int = 3, 
    warp_strength: float = 0.1,
    seed: int = None,
) -> np.ndarray:
    """
    时间扭曲增强函数
    Args:
        signal: 1d time series
        num_knots: 控制点数量
        warp_strength: 时间扭曲强度，0-1，默认0.1

    Returns:
        1d time series
    """
    signal = np.array(signal, dtype=np.float32)
    length = len(signal)
    orig_steps = np.arange(length)
    
    if seed is not None:
        np.random.seed(seed)

    while True:
        # 1. 随机选取控制点 (必须包含起点 0 和终点 length-1)。num_knots是不包含起点和终点的点数
        knots = np.linspace(0, 999, num=num_knots+2)
        
        # 2. 为控制点生成随机的时间偏移
        # 这里的偏移量决定了时间轴是被“拉伸”还是“压缩”
        random_offsets = np.random.uniform(-warp_strength, warp_strength, size=num_knots)
        offsets = np.concatenate(([0], random_offsets, [0]))
        
        # 3. 计算扭曲后的新时间轴位置
        warped_steps = knots + offsets * length
        
        # 确保新时间轴是严格单调递增的，防止时间倒流导致插值报错
        if np.all(np.diff(warped_steps) > 0):
            break
    
    
    # 4. 我们创建一个从“原始信号值”映射到“扭曲时间”的插值函数。
    f = PchipInterpolator(knots, warped_steps, extrapolate=True)
    
    # 5. 在标准的 0~999 时间轴上取值，得到最终被扭曲的信号
    warped_inds = f(orig_steps)
        
    return signal[warped_inds.astype(np.int32)]

def signal_shift_augmentation(
    signal: np.ndarray, # 1d time series
    shift_range: float = 0.05,
    seed: int = None,
) -> np.ndarray:
    """
    signal强度平移增强函数
    Args:
        signal: 1d time series
        shift_range: signal强度平移范围，从-shift_range~shift_range均匀分布采样
        seed: 随机种子，默认None

    Returns:
        1d time series
    """
    signal = np.array(signal, dtype=np.float32)
    
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 随机选取时间平移量
    random_shift = np.random.uniform(-shift_range, shift_range)
    
    signal = signal + random_shift
    signal = np.clip(signal, 0, 1)
    return signal

def add_noise_augmentation(
    signal: np.ndarray, # 1d time series
    noise_std: float = 0.01,
    seed: int = None,
) -> np.ndarray:
    """
    增加噪声增强函数
    Args:
        signal: 1d time series
        noise_std: 噪声标准差，0-1，默认0.05
        seed: 随机种子，默认None

    Returns:
        1d time series
    """
    signal = np.array(signal, dtype=np.float32)
    
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 随机选取噪声
    random_noise = np.random.normal(0, noise_std, size=len(signal))
    
    signal = signal + random_noise
    signal = np.clip(signal, 0, 1).astype(np.float32)
    return signal
