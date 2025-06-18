import random
import numpy as np
import pandas as pd
import copy
from scipy import interpolate
from scipy.sparse import csr_matrix
import ot
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw, dtw_path
from typing import Dict, Optional, Tuple, Union, Literal, List
from . import tools as tl


def get_density_matrix_from_an_obj(
    obj: dict,
    att: str = 'signal',
    y_len: Optional[int] = 1000,
    x_len: Optional[int] = 1000,
    target: Literal['dna1', 'window', 'dna2'] = 'window',
) -> np.ndarray:
    """Generate density of signal of an obj.

       I0 -> 1

    y

       0  -> 0          1000
                   x

    Args:
        obj (dict): obj
        att (str, optional): attribute to extract. Defaults to 'signal'.
        y_len (Optional[int], optional): the height of matrix, indicates I/I0. Defaults to 1000.
        x_len (Optional[int], optional): the width of matrix, indicates signal length. Defaults to 1000.
        target (Literal['dna1', 'window', 'dna2']): extract this part of the read. Defaults to 'window'.

    Returns:
        np.ndarray: density matrix. The first row indicates I0, the last indicates current with 0.
    """   
    X = tl.get_signals_for_reads_in_an_obj(obj, att, down_sample_to=x_len, normalize_by_openpore=True, target=target)
    den_arr = get_density_matrix_from_X(X, y_len=y_len)
    return den_arr

def get_density_matrix_from_X(
    X: np.ndarray,
    y_len: Optional[int] = 100,
) -> np.ndarray:
    """get density matrix from X.

       I0 -> 1

    y

       0  -> 0          100
                   x

    Args:
        X (np.ndarray): input signals, 2d array . Each row represents a read. Each read contains a series of signals.
        y_len (Optional[int], optional): split 0-1 into this number sections. Defaults to 100.

    Returns:
        np.ndarray: density matrix, 2d array.
    """
    X = np.array(X, dtype=np.float32)
    X[X<0] = 0
    X = y_len - 1 - np.round(X*y_len).astype(np.int16)
    X[X<0] = 0
    x_len = X.shape[1]
    data = np.ones(X.size)
    ys = X.flatten()
    xs = list(range(x_len)) * len(X)
    den_arr = csr_matrix((data, (ys, xs)), shape=(y_len, x_len)).toarray()
    den_arr = den_arr/den_arr.sum(axis=0)
    return den_arr




def get_ot_distance_betwen_two_densities(
    den1: np.ndarray,
    den2: np.ndarray,
    min_cutoff: float = 0.0,
    agg_to_mean: bool = False,
) -> Union[np.ndarray, float]:
    """Get OT distance between two densities.

    Args:
        den1 (np.ndarray): density matrix 1
        den2 (np.ndarray): density matrix 2
        min_cutoff (float, optional): the minimum value of density to be considered. Defaults to 0.0.
        agg_to_mean (bool, optional): whether to aggregate the OT distance to the mean. Defaults to False.

    Returns:    
        Union[np.ndarray, float]: OT distance
    """                   
    x1 = np.arange(den1.shape[1])[:,None]
    x2 = np.arange(den2.shape[1])[:,None]
    M = ot.dist(x1, x2, 'euclidean')

    distances = np.zeros(den1.shape[1])
    for i in range(den1.shape[1]):
        try:
            a = den1[:,i].copy()
            b = den2[:,i].copy()
            a[a<=min_cutoff] = 0
            b[b<=min_cutoff] = 0

            a = a / a.sum()
            b = b / b.sum()
        except:
            a = den1[:,i].copy()
            b = den2[:,i].copy()
            a = a / a.sum()
            b = b / b.sum()
        finally:
            distances[i] = ot.emd2(a=a, b=b, M=M)
    
    if agg_to_mean:
        return distances.mean()
    else:
        return distances


def get_consensus_signal_by_softdtw_barycenter_for_an_obj(
    obj: dict,
    n_time: int = 5,
    read_num_for_each_time: int = 2000,
    down_sample_to: int = 100,
    gamma: float = 0.001,
    att: str = 'signal',
    re_smooth: bool = True,
    smooth_att: str = 'smoothed_signal',
) -> np.array:
    """Get consensus signal by softdtw barycenter for an object.

    Args:
        obj (dict): object containing the data
        n_time (int, optional): number of time to randomly select data. Defaults to 5.
        read_num_for_each_time (int, optional): number of reads for each time. Defaults to 2000.
        down_sample_to (int, optional): down sample to this number. Defaults to 100.
        gamma (float, optional): gamma for softdtw. Defaults to 0.001.
        att (str, optional): attribute to extract. Defaults to 'signal'.
            if `re_smooth` is True, use `att` to do re_smooth.
        re_smooth (bool, optional): whether to smooth the signal. Defaults to True.
            if False, use `smooth_att` to find consensus.
            if True, smooth the signal and set `smooth_att` and use `smooth_att` to find consensus.
        smooth_att (str, optional): attribute to save the smoothed signal. Defaults to 'smoothed_signal'.

    Returns:
        np.array: consensus signal, down_sample_to * 2, 2d array
    """
    if re_smooth:
        tl.smooth_signal_by_median_filter_for_an_obj(obj, in_place=True, att=att, new_att=smooth_att)

    barycenters = []
    for i in range(n_time):
        sub_obj = tl.extract_x_reads_randomly(obj, read_num=read_num_for_each_time, seed=i)
        X = tl.get_signals_for_reads_in_an_obj(obj=sub_obj, down_sample_to=down_sample_to, 
                                               att=smooth_att, target='window', 
                                               normalize_by_openpore=True)
        barycenter = softdtw_barycenter(X, gamma=gamma)
        barycenters.append(barycenter)

    # 将获得的n_time次的barycenter合并成一个
    barycenter = softdtw_barycenter(np.concatenate(barycenters, axis=1).T, gamma=gamma)
    return barycenter
    

def align_signals_in_an_obj_to_a_consensus(
    obj: dict,
    consensus: np.array,
    att: str = 'smoothed_signal',
    reconstruct_from_att: str = 'signal',
    target: str = 'window',
    down_sample_to: int = 100,
    normalize_by_openpore: bool = True,
) -> np.array:
    """Align signals in an obj to a consensus and generate new aligned signals.
    Args:
        obj (dict): object containing the data
        consensus (np.array): consensus signal, down_sample_to * 1, 2d array
        att (str, optional): attribute to align. Defaults to 'smoothed_signal'.
        reconstruct_from_att (str, optional): use this attribute and alignment relationship to reconstruct the aligned signals. Defaults to 'signal'.
        target (str, optional): target to align. Defaults to 'window'.
        down_sample_to (int, optional): down sample to this number. Defaults to 100.
        normalize_by_openpore (bool, optional): normalize by openpore. Defaults to True.
    Returns:
        np.array: aligned signals, read_num * down_sample_to, 2d array 
    """
    X = tl.get_signals_for_reads_in_an_obj(obj=obj, down_sample_to=down_sample_to, 
                                               att=att, target=target, 
                                               normalize_by_openpore=normalize_by_openpore)
    X_raw = tl.get_signals_for_reads_in_an_obj(obj=obj, down_sample_to=down_sample_to, 
                                               att=reconstruct_from_att, target=target, 
                                               normalize_by_openpore=normalize_by_openpore)
    X_aligned = []
    for i in range(len(X)):
        s2 = X[i:i+1].T
        s2_raw = X_raw[i:i+1].T
        path, dtw_score = dtw_path(consensus, s2, global_constraint="sakoe_chiba", sakoe_chiba_radius=down_sample_to//10)
        s2_new = {}
        for i in path:
            (s, e) = i
            if s not in s2_new:
                s2_new[s] = []
            s2_new[s].append(s2_raw[e])
        for k, v in s2_new.items():
            s2_new[k] = np.mean(v)
        X_aligned.append(list(s2_new.values()))
    X_aligned = np.array(X_aligned)
    return X_aligned


        
        

def get_llikelihood_between_reads_of_an_obj_to_density_matrix(
    obj: dict,
    density_matrix: np.ndarray,
    y_len: int = 1000,
    x_len: int = 1000,
    eps: float = 1e-5,
    target: Literal['dna1', 'window', 'dna2'] = 'window',
    att: str = 'signal',
) -> pd.DataFrame:
    all_read_ids = list(obj.keys())
    X = tl.get_signals_for_reads_in_an_obj(obj, att, down_sample_to=x_len, normalize_by_openpore=True, target=target)
    X = np.array(X, dtype=np.float32)
    X = y_len - 1 - np.round(X*y_len).astype(np.int16)
    X[X<0] = 0
    llikelihood = _get_llikelihood_between_density_matrix_and_dealed_signals(density_matrix=density_matrix, signals=X, eps=eps)
    llikelihood_df = pd.DataFrame({'llikelihood': llikelihood}, index=all_read_ids)
    return llikelihood_df

def _get_llikelihood_between_density_matrix_and_dealed_signals(
    density_matrix: np.ndarray,
    signals: np.ndarray,
    eps: Optional[float] = 1e-5,
):
    llikelihood = _get_llikelihood_between_density_matrix_and_case_signal(density_matrix=density_matrix, case_signal=signals, eps=eps, axis=1)
    return llikelihood 

def _get_llikelihood_between_density_matrix_and_case_signal(
    density_matrix: np.ndarray,
    case_signal: np.ndarray,
    axis: Literal[0, 1],
    eps: Optional[float] = 1e-5,
):
    dens = density_matrix[case_signal,range(density_matrix.shape[1])]
    dens[dens==0] = eps
    llikelihood = np.sum(np.log(dens), axis=axis)
    return llikelihood

    