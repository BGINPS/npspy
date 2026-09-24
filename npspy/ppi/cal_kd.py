from audioop import lin2adpcm
from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def binding_model(X, Bmax, Kd, NS, Background):
    """
    非线性结合模型：Y = Bmax*X/(Kd + X) + NS*X + Background
    """
    return (Bmax * X) / (Kd + X) + NS * X + Background

def binding_model_simple(X, Bmax, Kd):
    return (Bmax * X) / (Kd + X)

def fit_binding_data_positive(X_data, Y_data):
    """
    拟合数据，并强制限制所有参数 > 0
    """
    X_data = np.array(X_data, dtype=float)
    Y_data = np.array(Y_data, dtype=float)

    # 1. 设置初始猜测值 (p0)
    # 好的初始猜测能帮助算法更快收敛
    Bmax_guess = max(Y_data) - min(Y_data)
    Kd_guess = np.median(X_data)
    NS_guess = 0.1
    Background_guess = min(Y_data)
    p0 = [Bmax_guess, Kd_guess, NS_guess, Background_guess]

    n = len(Y_data) # 样本量
    p = 4           # 参数个数 (Bmax, Kd, NS, Background)

    # 2. 设置边界 (bounds)
    # 下界全是 0 (保证正数)，上界全是无穷大
    # bounds=( [Bmax_min, Kd_min, NS_min, Background_min], [Bmax_max, ...] )
    lower_bounds = [0, 0, 0, 0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    try:
        # 3. 执行拟合
        # 注意：一旦设置了 bounds，curve_fit 会自动从 'lm' 算法切换到 'trf' 算法
        popt, pcov = curve_fit(
            binding_model, 
            X_data, 
            Y_data, 
            p0=p0, 
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000  # 增加最大迭代次数以防不收敛
        )
        
        Bmax, Kd, NS, Background = popt

        # 3. 计算预测值与残差
        Y_pred = binding_model(X_data, Bmax, Kd, NS, Background)
        residuals = Y_data - Y_pred
        
        # --- 核心修改：计算 P 值相关指标 ---
        
        # A. 计算 R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y_data - np.mean(Y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # B. 计算整体模型的 F 检验 P 值
        # F = (解释的方差 / 参数自由度) / (未解释的方差 / 残差自由度)
        # 注意：这里简化处理，假设模型至少包含截距效应，对比的是常数模型
        if ss_res == 0:
            f_statistic = np.inf
            p_value_model = 0
        else:
            # 均方回归 (MSR) 近似
            msr = ((ss_tot - ss_res) / p)
            # 均方误差 (MSE)
            mse = (ss_res / (n - p))
            
            if mse == 0: mse = 1e-10 # 防止除以0
            
            f_statistic = msr / mse
            p_value_model = stats.f.sf(f_statistic, p, n - p)

        return {
            'Bmax': Bmax,
            'Kd': Kd,
            'NS': NS,
            'Background': Background,
            'success': True,
            'r2': r_squared,
            'p_value_model': p_value_model,
        }

    except RuntimeError as e:
        print("拟合失败:", e)
        return {'success': False}

def fit_binding_data_positive_simple(X_data, Y_data):
    """
    拟合数据，并强制限制所有参数 > 0
    """
    X_data = np.array(X_data, dtype=float)
    Y_data = np.array(Y_data, dtype=float)

    # 1. 设置初始猜测值 (p0)
    # 好的初始猜测能帮助算法更快收敛
    Bmax_guess = max(Y_data) - min(Y_data)
    Kd_guess = np.median(X_data)
    p0 = [Bmax_guess, Kd_guess]

    n = len(Y_data) # 样本量
    p = 2           # 参数个数 

    # 2. 设置边界 (bounds)
    lower_bounds = [0, 0,]
    upper_bounds = [100, np.inf]

    try:
        # 3. 执行拟合
        # 注意：一旦设置了 bounds，curve_fit 会自动从 'lm' 算法切换到 'trf' 算法
        popt, pcov = curve_fit(
            binding_model_simple, 
            X_data, 
            Y_data, 
            p0=p0, 
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000  # 增加最大迭代次数以防不收敛
        )
        
        Bmax, Kd, = popt

        # 3. 计算预测值与残差
        Y_pred = binding_model_simple(X_data, Bmax, Kd)
        residuals = Y_data - Y_pred
        
        # --- 核心修改：计算 P 值相关指标 ---
        
        # A. 计算 R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y_data - np.mean(Y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # B. 计算整体模型的 F 检验 P 值
        # F = (解释的方差 / 参数自由度) / (未解释的方差 / 残差自由度)
        # 注意：这里简化处理，假设模型至少包含截距效应，对比的是常数模型
        if ss_res == 0:
            f_statistic = np.inf
            p_value_model = 0
        else:
            # 均方回归 (MSR) 近似
            msr = ((ss_tot - ss_res) / p)
            # 均方误差 (MSE)
            mse = (ss_res / (n - p))
            
            if mse == 0: mse = 1e-10 # 防止除以0
            
            f_statistic = msr / mse
            p_value_model = stats.f.sf(f_statistic, p, n - p)

        return {
            'Bmax': Bmax,
            'Kd': Kd,
            'success': True,
            'r2': r_squared,
            'p_value_model': p_value_model
        }

    except RuntimeError as e:
        print("拟合失败:", e)
        return {'success': False}


def plot_formula(ax, formula, x_range=(-10, 10), num_points=1000, xlabel="concentration (nM)", ylabel="bound ratio", color='red', label='test'):
    """
    根据给定的公式字符串绘制 y = f(x) 的曲线。

    参数:
        formula: 字符串，表示 y 关于 x 的公式，例如 "x**2", "np.sin(x)", "2*x + 1"
        x_range: 元组，x 的取值范围 (min, max)
        num_points: 整数，采样点数
        title: 图像标题
        xlabel: x轴标签
        ylabel: y轴标签
    """
    # 生成 x 数据
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # 动态计算 y 值
    try:
        y = eval(formula)
    except Exception as e:
        print("公式计算错误:", e)
        return

    # 绘图
    ax.plot(x, y, color=color, linewidth=2, linestyle='--', label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.grid(True, alpha=0.3)
