#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: denoise.py
@Description: description of this file
@Datatime: 2025/06/16 14:50:12
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import t

from . import tools as tl

def bayesian_variance_inference(
    data: np.ndarray,         # 1-dimensional array of observed data
    known_mean: float,         # Known population mean μ
    prior_alpha: float = 3.0,  # Alpha parameter for inverse gamma prior
    prior_beta: float = 2.0    # Beta parameter for inverse gamma prior
) -> Tuple[float, float]:
    """
    Bayesian inference for posterior distribution of variance in a normal distribution.
    
    Assumes data follows X ~ N(μ, σ²) with known mean μ and unknown variance σ².
    Uses conjugate inverse gamma prior for σ²: IG(α, β)
    Returns posterior parameters for inverse gamma distribution: IG(α_post, β_post)
    
    Args:
    data : np.ndarray       -- Observed data samples (1D array)
    known_mean : float      -- Known population mean μ
    prior_alpha : float     -- Prior shape parameter α (default: 3.0)
    prior_beta : float      -- Prior scale parameter β (default: 2.0)
    
    Returns:
    post_alpha, post_beta : Tuple[float, float] 
        -- Parameters of posterior inverse gamma distribution
    """
    n = len(data)
    
    # Compute sum of squared deviations from known mean
    sum_squared = np.sum((data - known_mean)**2)
    
    # Calculate posterior parameters using conjugate prior update
    post_alpha = prior_alpha + n / 2
    post_beta = prior_beta + 0.5 * sum_squared
    
    return post_alpha, post_beta


def get_post_parameters_for_an_obj(
    obj: dict,
    prior_alpha: float = 3.0, 
    prior_beta: float = 2.0,
    label: str = 'posterior',
) -> pd.DataFrame:
    """
    Calculate the mean and posterior alpha and beta parameters for a given object.

    Args:
        obj (dict): A dictionary where keys are read IDs and values are read objects containing 'signal' and 'window'.
        prior_alpha (float): Prior alpha parameter for the inverse gamma distribution.
        prior_beta (float): Prior beta parameter for the inverse gamma distribution.
        label (str): Label for the resulting DataFrame index.
    Returns:
        pd.DataFrame: A DataFrame with mean and posterior alpha and beta parameters.
    """
    data, all_data = [], []
    for read_id, read_obj in obj.items():
        if 'signal' not in read_obj or 'window' not in read_obj:
            raise ValueError(f"Read {read_id} does not contain 'signal' or 'window'.")
        if not isinstance(read_obj['window'], (list, tuple)) or len(read_obj['window']) != 2:
            raise ValueError(f"The 'window' for read {read_id} must be a list or tuple of length 2.")
        diff_signal = tl.get_diff_for_a_read_obj(read_obj, only_window=True)
        data.append(diff_signal[len(diff_signal)//2])
        all_data.extend(diff_signal)
    data = np.array(data, dtype=np.float32)
    known_mean = np.mean(all_data)

    post_alpha, post_beta = bayesian_variance_inference(
        data, 
        known_mean=known_mean, 
        prior_alpha=prior_alpha, 
        prior_beta=prior_beta
    )

    df = pd.DataFrame(
        {
            'mean': [known_mean],
            'post_alpha': [post_alpha],
            'post_beta': [post_beta],
        },
        index=[label]
    )

    return df


def get_post_parameters_for_objs(
    objs: List[dict],
    prior_alpha: float = 3.0, 
    prior_beta: float = 2.0,
    labels: List[str] = None,
) -> pd.DataFrame:
    """
    Calculate the mean and posterior alpha and beta parameters for multiple objects.

    Args:
        objs (List[dict]): A list of dictionaries where each dictionary contains read IDs and their corresponding read objects.
        prior_alpha (float): Prior alpha parameter for the inverse gamma distribution.
        prior_beta (float): Prior beta parameter for the inverse gamma distribution.
        labels (List[str]): Optional list of labels for the resulting DataFrame index. If None, default labels will be used.
    Returns:
        pd.DataFrame: A DataFrame with mean and posterior alpha and beta parameters for each object.
    """
    if labels is None:
        labels = [f'posterior_{i}' for i in range(len(objs))]
    assert len(objs) == len(labels), "Length of objs and labels must be the same."

    df = []
    for obj, lable in zip(objs, labels):
        df_ = get_post_parameters_for_an_obj(
            obj=obj,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            label=lable,
        )
        df.append(df_)
    df = pd.concat(df, axis=0)
    df.index.name = 'label'
    return df



def marginal_pdf_via_t(
    x: np.ndarray, # 1d 
    mu: float, 
    alpha: float, 
    beta: float,
) -> np.ndarray:
    """
    Calculate marginal probability density via t-distribution

    Args:
        x (np.array): observed value(s)
        mu (float): known mean of normal distribution
        alpha (float): shape parameter of inverse gamma distribution
        beta (float): scale parameter of inverse gamma distribution
    Returns:
        np.ndarray: marginal probability density value(s)
    """

    df = 2 * alpha                   # degrees of freedom
    scale = np.sqrt(beta / alpha)    # scale parameter
    return t.pdf(x, df=df, loc=mu, scale=scale)

def marginal_sum_of_loglikelihood_via_t(
    x: np.ndarray, # 1d
    mu: float, 
    alpha: float, 
    beta: float,
) -> float:
    """
    Calculate the sum of log likelihood via t-distribution

    Args:
        x (np.ndarray): observed value(s)
        mu (float): known mean of normal distribution
        alpha (float): shape parameter of inverse gamma distribution
        beta (float): scale parameter of inverse gamma distribution
    Returns:
        float: sum of log likelihood value(s)
    """
    pdf_values = marginal_pdf_via_t(x, mu, alpha, beta)
    return np.sum(np.log(pdf_values))



def get_sum_of_loglikelihood_for_a_read_obj(
    read_obj: dict,
    post_params_df: pd.DataFrame,
    high_quality_index: str = 'hq',
    low_quality_index: str = 'lq',
) -> pd.DataFrame:
    """
    Calculate the sum of log likelihood for a given read object using posterior parameters.
    Args:
        read_obj (dict): A dictionary containing the read object with keys 'signal' and 'window'.
        post_params_df (pd.DataFrame): DataFrame containing posterior parameters with columns 'mean', 'post_alpha', and 'post_beta'.
        high_quality_index (str): Index for high quality model parameters in post_params_df.
        low_quality_index (str): Index for low quality model parameters in post_params_df.
    Returns:
        pd.DataFrame: A DataFrame with the sum of log likelihood for high and low quality models.
    """
    diff_signal = tl.get_diff_for_a_read_obj(
        read_obj=read_obj,
        only_window=True,
    )

    sum_of_log_L_for_high_qaulity_model = marginal_sum_of_loglikelihood_via_t(
        diff_signal,
        mu=post_params_df.loc[high_quality_index, 'mean'],
        alpha=post_params_df.loc[high_quality_index, 'post_alpha'],
        beta=post_params_df.loc[high_quality_index, 'post_beta'],
    )

    sum_of_log_L_for_low_qaulity_model = marginal_sum_of_loglikelihood_via_t(
        diff_signal,
        mu=post_params_df.loc[low_quality_index, 'mean'],
        alpha=post_params_df.loc[low_quality_index, 'post_alpha'],
        beta=post_params_df.loc[low_quality_index, 'post_beta'],
    )

    return pd.DataFrame(
        {
            'sum_of_log_L_for_high_quality_model': [sum_of_log_L_for_high_qaulity_model],
            'sum_of_log_L_for_low_quality_model': [sum_of_log_L_for_low_qaulity_model],
        }
    )

def get_sum_of_loglikelihood_for_an_obj(
    obj: dict,
    post_params_df: pd.DataFrame,
    high_quality_index: str = 'hq',
    low_quality_index: str = 'lq',
) -> pd.DataFrame:
    """
    Calculate the sum of log likelihood for each read in an object using posterior parameters.
    
    Args:
        obj (dict): A dictionary where keys are read IDs and values are read objects containing 'signal' and 'window'.
        post_params_df (pd.DataFrame): DataFrame containing posterior parameters with columns 'mean', 'post_alpha', and 'post_beta'.
        high_quality_index (str): Index for high quality model parameters in post_params_df.
        low_quality_index (str): Index for low quality model parameters in post_params_df.
    
    Returns:
        pd.DataFrame: A DataFrame with read IDs as index and columns for sum of log likelihood for high and low quality models.
    """
    df = []
    for read_id, read_obj in obj.items():
        log_likelihoods = get_sum_of_loglikelihood_for_a_read_obj(
            read_obj=read_obj,
            post_params_df=post_params_df,
            high_quality_index=high_quality_index,
            low_quality_index=low_quality_index,
        )
        log_likelihoods.index = [read_id]
        df.append(log_likelihoods)
    df = pd.concat(df, axis=0)
    return df


def split_an_obj_as_high_and_low_quality(
    obj: dict,
    sum_of_log_likelihood_df: pd.DataFrame,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Split an object into high and low quality based on sum of log likelihood.

    Args:
        obj (dict): A dictionary where keys are read IDs and values are read objects.
        sum_of_log_likelihood_df (pd.DataFrame): DataFrame with sum of log likelihood for each read.
    
    Returns:
        Tuple[Dict[str, dict], Dict[str, dict]]: Two dictionaries containing high quality and low quality reads.
    """
    high_quality_obj = {}
    low_quality_obj = {}
    
    for read_id, read_obj in obj.items():
        if sum_of_log_likelihood_df.loc[read_id, 'sum_of_log_L_for_high_quality_model'] > \
           sum_of_log_likelihood_df.loc[read_id, 'sum_of_log_L_for_low_quality_model']:
            high_quality_obj[read_id] = read_obj
        else:
            low_quality_obj[read_id] = read_obj
    
    return high_quality_obj, low_quality_obj