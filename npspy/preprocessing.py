import random
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Literal, List
from . import tools as tl
from sklearn.model_selection import train_test_split
from . import io
from . import density as dn
import copy
import time
from scipy import signal
import os
import re


def filter_out_reads_without_widows(
    obj,
):
    """Filter out the reads with no target windows.

    Args:
        obj (_type_): the obj to be filtered
    """
    read_ids_with_no_windows = [read_id for read_id, read_obj in obj.items() if read_obj['window']==None]
    tl.delete_reads_in_an_obj(obj=obj, reads_need_to_remove=read_ids_with_no_windows, in_place=True)
    


def filter_out_reads_with_too_short_or_too_long_total_length(
    obj_dict,
    sd_fold: float = 1.0,
):
    """Filter out reads by total length. First calculate mean and sd of total length.
    Then keep reads with length within the range of mean-sd_fold*sd to mean+sd_fold*sd.

    Args:
        obj_dict (_type_): obj
        sd_fold (float): the fold of sd, within this range, would be kept
    """
    filter_out_reads_by_sd_of_length(obj=obj_dict, by='total_length', sd_fold=sd_fold)


def filter_out_reads_with_too_short_or_too_long_window_length(
    obj_dict,
    sd_fold: float = 1.0,
    min_len: int = 1000,
):
    """Filter out reads by window length. First calculate mean and sd of total length.
    Then keep reads with length within the range of mean-sd_fold*sd to mean+sd_fold*sd.

    Args:
        obj_dict (_type_): obj
        sd_fold (float): the fold of sd, within this range, would be kept
        min_len (int): windows with length less than this value would be removed
    """
    filter_out_reads_by_sd_of_length(obj=obj_dict, by='window_length', sd_fold=sd_fold, min_len=min_len)


def filter_out_reads_by_sd_of_length(
    obj: dict,
    by: Literal['window_length', 'total_length'] = 'window_length',
    sd_fold: float = 1.0,
    min_len: int = 0,
):
    if by.lower() not in ['window_length', 'total_length']:
        raise Exception(f"Unknown by `{by}`")
    if by == 'window_length':
        length = tl.window_length_for_one_obj(obj)
    elif by == 'total_length':
        length = tl.total_length_for_one_obj(obj)
    short, long = max(0, length.mean() - length.std() * sd_fold), length.mean() + length.std() * sd_fold
    short = max(min_len, short)
    read_ids_with_abnormal_length = length[(length<short) | (length>long)].index
    tl.delete_reads_in_an_obj(obj_dict=obj, read_ids_to_be_removed=read_ids_with_abnormal_length)


def filter_out_reads_with_negative_signals_in_window(
    obj,
):
    read_ids_with_negtive_signals = [read_id for read_id, read_obj in obj.items() if np.any(read_obj['signal'][read_obj['window'][0]:read_obj['window'][1]]<0)]
    tl.delete_reads_in_an_obj(obj=obj,  reads_need_to_remove=read_ids_with_negtive_signals, in_place=True)

def filter_out_reads_with_curr_high_than_i0_in_window(
    obj: dict,
):
    read_ids_with_signal_higher_than_i0 = [read_id for read_id, read_obj in obj.items() if np.any(read_obj['signal'][read_obj['window'][0]:read_obj['window'][1]]>read_obj['OpenPore'])]
    tl.delete_reads_in_an_obj(obj=obj,  reads_need_to_remove=read_ids_with_signal_higher_than_i0, in_place=True)


def clean_obj(
    obj: dict,
    sd_fold: float = 1.0,
    clean_by_label: bool = True,
    direct_oprate_on_obj: bool = False,
) -> (dict, dict):
    """Clean obj. keep reads within mean+-1*sd of pd2rd, i2i0_mean, i2i0_std

    Args:
        obj (dict): an obj
        sd_fold (float, optional): the fold of sd of some attributes, within this range (mean-sd_fold*sd to mean+sd_fold*sd),
                                    would be kept. Defaults to 1.0.

        clean_by_label (boo, optional): whether to clean obj seperatelly for each class or clean one time together.
    
    Returns:
        dict: obj, cutoffs_dict
    """
    if not direct_oprate_on_obj:
        obj = copy.deepcopy(obj)
    tl.state_obj(obj=obj, des='before clean')
    objs_for_each_label = []
    all_labels = ['label']
    if clean_by_label:
        all_labels = np.unique(np.array([read_obj['label'] for read_id, read_obj in obj.items()]))
        for label in all_labels:
            objs_for_each_label.append(extract_reads_with_labels(obj=obj, labels=[label]))
    else:
        objs_for_each_label.append(obj)
    
    clean_objs = []
    cutoffs_dict = {}
    for indx, one_obj in enumerate(objs_for_each_label):
        if clean_by_label:
            print(all_labels[indx])
        pd2rd_left, pd2rd_right = find_sd_left_right_cutoff_for_an_att(obj=one_obj, att='pd2rd', sd_fold=sd_fold)
        window_i2i0_mean_left, window_i2i0_mean_right = find_sd_left_right_cutoff_for_an_att(obj=one_obj, att='mean_of_I/I0', sd_fold=sd_fold)
        window_i2i0_std_left, window_i2i0_std_right = find_sd_left_right_cutoff_for_an_att(obj=one_obj, att='std_of_I/I0', sd_fold=sd_fold)
        cutoffs_dict[all_labels[indx]] = {'pd2rd_left': pd2rd_left, 'pd2rd_right': pd2rd_right,
                                          'window_i2i0_mean_left': window_i2i0_mean_left, 'window_i2i0_mean_right': window_i2i0_mean_right,
                                          'window_i2i0_std_left': window_i2i0_std_left, 'window_i2i0_std_right': window_i2i0_std_right}
        print(f'pd2rd_left: {pd2rd_left}\npd2rd_right: {pd2rd_right}\nwindow_i2i0_mean_left: {window_i2i0_mean_left}\nwindow_i2i0_mean_right: {window_i2i0_mean_right}\nwindow_i2i0_std_left: {window_i2i0_std_left}\nwindow_i2i0_std_right: {window_i2i0_std_right}')
        select_reads_within_att_range(obj=one_obj, att='pd2rd', left=pd2rd_left, right=pd2rd_right)
        select_reads_within_att_range(obj=one_obj, att='mean_of_I/I0', left=window_i2i0_mean_left, right=window_i2i0_mean_right)
        select_reads_within_att_range(obj=one_obj, att='std_of_I/I0', left=window_i2i0_std_left, right=window_i2i0_std_right)
        clean_objs.append(one_obj)

    obj = tl.combine_objs(obj_list=clean_objs)
    tl.state_obj(obj=obj, des='after filtering out reads by std of pd2rd, window_i2i0_mean, window_i2i0_std') 

    return obj, cutoffs_dict


def filter_out_reads_by_window_len(
    obj: dict,
    min_len: int = 1000,
):
    reads_need_to_be_removed = []
    for read_id, read_obj in obj.items():
        if read_obj['window_length'] < min_len:
            reads_need_to_be_removed.append(read_id)
    tl.delete_reads_in_an_obj(obj=obj, reads_need_to_remove=reads_need_to_be_removed, in_place=True)


def filter_out_reads_by_dna1_len(
    obj: dict,
    dna1_min_len: int = 8900,
    dna1_max_len: int = 12900,
):
    reads_need_to_be_removed = []
    for read_id, read_obj in obj.items():
        discard_by_dna1 = read_obj['dna1_len'] < dna1_min_len or read_obj['dna1_len'] > dna1_max_len
        if discard_by_dna1:
            reads_need_to_be_removed.append(read_id)
    tl.delete_reads_in_an_obj(obj=obj, reads_need_to_remove=reads_need_to_be_removed, in_place=True)


def filter_out_reads_by_dna2_len(
    obj: dict,
    dna2_min_len: int = 7800,
    dna2_max_len: int = 12200,
):
    reads_need_to_be_removed = []
    for read_id, read_obj in obj.items():
        discard_by_dna2 = read_obj['dna2_len'] < dna2_min_len or read_obj['dna2_len'] > dna2_max_len
        if discard_by_dna2:
            reads_need_to_be_removed.append(read_id)
    tl.delete_reads_in_an_obj(obj=obj, reads_need_to_remove=reads_need_to_be_removed, in_place=True)

    


def select_reads_within_att_range(
    obj: dict,
    att: str,
    left: float,
    right: float,
):
    reads_need_to_be_removed = []
    for read_id, read_obj in obj.items():
        if read_obj[att]<left or read_obj[att]>right:
            reads_need_to_be_removed.append(read_id)
    tl.delete_reads_in_an_obj(obj=obj, reads_need_to_remove=reads_need_to_be_removed, in_place=True)

    
    

def find_sd_left_right_cutoff_for_an_att(
    obj: dict,
    att: str,
    sd_fold: float = 1.0,
):
    att_values = []
    for read_id, read_obj in obj.items():
        att_values.append(read_obj[att])
    att_values = np.array(att_values)
    mu, std = np.mean(att_values), np.std(att_values)
    left, right = mu-std*sd_fold, mu+std*sd_fold
    return left, right

    


# def filter_out_invalid_reads_and_add_window_att(
#     obj: dict,
#     min_len: int = 1000,
# ):
#     """1. filter out reads without windows
#     2. filter out reads with window length shorter than min_len
#     3. filter out reads with negative signals
#     4. filter out reads with current larger than I0
#     5. add attributes of windows to reads

#     Args:
#         obj (dict): obj
#         min_len (int, optional): reads with window length shorter than this value would be discarded. Defaults to 1000.
#     """
#     add_read_att(obj)
#     filter_out_invalid_reads(obj, min_len=min_len)



# def filter_out_invalid_reads(
#     obj: dict,
#     min_len: int = 1000,
# ):
#     """1. filter out reads without windows
#     2. filter out reads with window length shorter than min_len
#     3. filter out reads with negative signals
#     4. filter out reads with current larger than I0

#     Args:
#         obj (dict): obj
#         min_len (int, optional): reads with window length shorter than this value would be discarded. Defaults to 1000.
#     """
#     filter_out_reads_without_widows(obj)
#     tl.state_obj(obj=obj, des='after filtering out reads without windows')
#     filter_out_reads_by_window_len(obj, min_len=min_len)
#     tl.state_obj(obj=obj, des=f'after filtering out reads with window length shorter than {min_len}') 
#     filter_out_reads_with_negative_signals_in_window(obj)
#     tl.state_obj(obj=obj, des='after filtering out reads with negative signals')
#     filter_out_reads_with_curr_high_than_i0_in_window(obj)
#     tl.state_obj(obj=obj, des='after filtering out reads with current larger than I0')

# def add_read_att(
#     obj: dict,
#     atts: list = ['window_length', 'pd2rd', 'window_i2i0_mean', 'window_i2i0_std', 'dna1_len', 'dna2_len']
# ):
#     for read_id, read_obj in obj.items():
#         window_start, window_end = tl.get_window_start_end_for_one_read(read_obj)
#         if window_start == None:
#             for att in atts:
#                 read_obj[att] = None
#             continue
    
#         window_signals = tl.extract_signal_in_window_for_one_read_obj(read_obj=read_obj)

#         if 'window_length' in atts:
#             read_obj['window_length'] = len(window_signals)
#         if 'pd2rd' in atts:
#             read_obj['pd2rd'] = len(window_signals)/len(read_obj['signal'])
#         if 'window_i2i0_mean' in atts:
#             window_signals_i2i0 = window_signals/read_obj['openpore']
#             read_obj['window_i2i0_mean'] = np.mean(window_signals_i2i0)
#         if 'window_i2i0_std' in atts:
#             window_signals_i2i0 = (window_signals/read_obj['openpore']).astype(np.float64)
#             read_obj['window_i2i0_std'] = np.std(window_signals_i2i0)
#         if 'dna1_len' in atts:
#             read_obj['dna1_len'] = window_start
#         if 'dna2_len' in atts:
#             read_obj['dna2_len'] = len(read_obj['signal']) - window_end - 1
        




def _add_label(obj, label):
    assert isinstance(obj, dict)
    assert isinstance(label, str)
    for read_id, read_obj in obj.items():
        read_obj['label'] = label

def add_label(
    objs: Union[dict, list], 
    labels: Union[str, list],
):
    """Add label to each read in each obj

    Args:
        objs (Union[dict, list]): an obj or a list of objs
        labels (Union[str, list]): a label or a list of labels
    """
    if isinstance(objs, list) and isinstance(labels, list):
        assert len(objs) == len(labels)
        for obj, label in zip(objs, labels):
            _add_label(obj, label)
    else:
        _add_label(objs, labels)


# def get_signals_for_reads_in_an_obj(
#     obj_dict: dict, 
#     all_read_ids: list, 
#     scale_to: int = 1000, 
#     normalize_by_openpore: bool = True,
#     target: Literal['dna1', 'window', 'dna2'] = 'window',
# ) -> list:
#     """Extract signals in windows of an obj. signal would be dowm sampled or normalized.

#     Args:
#         obj_dict (dict): an obj
#         all_read_ids (list): all read ids in this obj
#         scale_to (int, optional): scale signal lenght to this value. Defaults to 1000.
#         normalize_by_openpore (bool, optional): normalized or not. Defaults to True.
#         target (Literal['dna1', 'window', 'dna2']): extract this part of the read. Defaults to 'window'.

#     Returns:
#         np.array: signals
#     """
#     assert target in ['dna1', 'window', 'dna2']
#     Xs = []
#     for read_id in all_read_ids:
#         read_obj = obj_dict[read_id]
#         start, end = read_obj['window'][0], read_obj['window'][1]
#         # start = max(start - 100, 0) # test
#         # end = end + 100             # test
#         if target == 'window':
#             target_start, target_end = start, end+1
#         elif target == 'dna1':
#             target_start, target_end = 0, start
#         elif target == 'dna2':
#             target_start, target_end = end+1, len(read_obj['signal'])

#         X = tl.down_sampling(read_obj['signal'][target_start:target_end], down_sample_to=scale_to)

#         if normalize_by_openpore:
#             X /= read_obj['openpore']
#         Xs.append(list(X))
#     return Xs


# def normalize_signals_in_an_obj_by_openpore(
#     obj,
#     inplace: bool = False
# ):
#     if not inplace:
#         obj = copy.deepcopy(obj)
#     for read_id, read_obj in obj.items():
#         read_obj['signal'] /= read_obj['openpore']
#     return obj

# def get_labels_in_windows(
#     obj_dict: dict, 
#     all_read_ids: list,
# ) -> list:
#     """Get labels for windows in an obj. If there is not lable in read, use "" as label.

#     Args:
#         obj_dict (dict): obj
#         all_read_ids (list): all read ids

#     Returns:
#         list: labels
#     """
#     if 'label' not in obj_dict[all_read_ids[0]]:
#         labels = ["" for i in all_read_ids]
#     else:
#         labels = [obj_dict[read_id]['label'] for read_id in all_read_ids]
    
#     return labels






    
def down_sample_obj_by_label(
    obj: dict,
    down_sample_to: int = 0,
    seed: int = 0,
    print_info: bool = True
):
    """Select a specific number (down_sample_to) of reads for all class.

    Args:
        obj (dict): dict
        down_sample_to (int, optional): select this number of reads for each class (label). If this number is larger than 
                                        the read number of the smallest class, set this number to the smallest number. 
                                        If set to 0, use the smallest number. Defaults to 0.
        seed (int, optional): seed. Defaults to 0.
    """
    df = pd.DataFrame([[read_id, read_obj['label']] for read_id, read_obj in obj.items()], columns=['read_id', 'label'])
    if down_sample_to == 0:
        down_sample_to = df['label'].value_counts().min()
    else:
        down_sample_to = min(df['label'].value_counts().min(), down_sample_to)
    if print_info:
        print(f"dowm sapmle read number of each class to {down_sample_to}")
    selected_read_ids = df.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(n=down_sample_to, random_state=seed))['read_id']
    need_to_remove_ids = obj.keys() - selected_read_ids
    tl.delete_reads_in_an_obj(obj, need_to_remove_ids)
    if print_info:
        tl.state_obj(obj=obj, des='after down sampling')



def split_obj_into_train_test_obj(
    obj: dict,
    sample_num_of_each_cluster_in_train: int,
    sample_num_of_each_cluster_in_test: int,
    lables_keep_in_test: list,
    seed: int = 0,
) -> (dict, dict):
    obj = copy.deepcopy(obj)
    if lables_keep_in_test:
        change_obj_labels_to_other(obj, lables_need_to_change=lables_keep_in_test)
    sample_num_for_classes = tl.state_obj(obj, des='before split', return_sample_num_for_classes=True)
    assert sample_num_of_each_cluster_in_train + sample_num_of_each_cluster_in_test <= sample_num_for_classes[sample_num_for_classes.index != 'other'].min()
    if lables_keep_in_test:
        assert sample_num_for_classes['other'] >= sample_num_of_each_cluster_in_test
    other_obj = extract_reads_with_labels(obj, labels=['other'])
    remove_reads_in_obj1_present_in_obj2(obj1=obj, obj2=other_obj)
    no_other_obj = obj

    if lables_keep_in_test:
        other_obj = tl.random_extract_reads(obj=other_obj, extract_num=sample_num_of_each_cluster_in_test, validated_read=False)
    down_sample_obj_by_label(obj=no_other_obj, down_sample_to=sample_num_of_each_cluster_in_train+sample_num_of_each_cluster_in_test, print_info=False, seed=seed)
    train_obj = copy.deepcopy(no_other_obj)
    down_sample_obj_by_label(obj=train_obj, down_sample_to=sample_num_of_each_cluster_in_train, print_info=False, seed=seed)
    remove_reads_in_obj1_present_in_obj2(obj1=no_other_obj, obj2=train_obj)
    test_obj = no_other_obj
    test_obj = tl.combine_objs(obj_list=[test_obj, other_obj])
    tl.state_obj(train_obj, des='training obj')
    tl.state_obj(test_obj, des='testing obj')
    return train_obj, test_obj


def split_obj_into_train_test_obj_by_ratio(
    obj: dict,
    test_ratio: float = 0.3,
    seed: int = 0,
) -> (dict, dict):
    df = pd.DataFrame([[read_id, read_obj['label']] for read_id, read_obj in obj.items()], columns=['read_id', 'label'])
    testdata_read_ids = df.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(frac=test_ratio, random_state=seed))['read_id']
    test_obj = extract_reads_as_an_obj(obj, read_ids=testdata_read_ids)
    traindata_read_ids = sorted(list(obj.keys() - testdata_read_ids))
    train_obj = extract_reads_as_an_obj(obj, read_ids=traindata_read_ids)
    return train_obj, test_obj

        





def extract_reads_as_an_obj(
    obj: dict,
    read_ids: list,
) -> dict:
    sub_obj = {}
    for read_id in read_ids:
        sub_obj[read_id] = obj[read_id]
    return sub_obj

    

def remove_reads_in_obj1_present_in_obj2(
    obj1: dict,
    obj2: dict,
):
    read_ids_both_in_two_objs = np.intersect1d(list(obj1.keys()), list(obj2.keys()))
    tl.delete_reads_in_an_obj(obj_dict=obj1, read_ids_to_be_removed=read_ids_both_in_two_objs)


def extract_reads_with_labels(
    obj: dict,
    labels: list,
) -> dict:
    """extract reads with labels

    Args:
        obj (dict): obj
        labels (list): labels

    Returns:
        dict: obj
    """
    obj = {read_id:read_obj for read_id, read_obj in obj.items() if read_obj['label'] in labels}
    return obj




def change_obj_labels_to_other(
    obj: dict,
    lables_need_to_change: list,
):
    """Change reads with label in lables_need_to_change to other

    Args:
        obj (dict): obj
        lables_need_to_change (list): if a read with label within this list, change it to other
    """
    for read_id, read_obj in obj.items():
        if read_obj['label'] in lables_need_to_change:
            read_obj['label'] = 'other'
        


def remove_reads_with_labels(
    obj: dict,
    labels: list,
):
    """Remove reads in an obj that have label within labels

    Args:
        obj (dict): obj
        labels (list): the labels need to be removed
    """
    obj_with_to_be_removed_labels = extract_reads_with_labels(obj, labels=labels)
    remove_reads_in_obj1_present_in_obj2(obj, obj_with_to_be_removed_labels)




# def filter_obj_recipe(
#     obj: dict,
#     out_dir: str,
#     prefix: str = None,
#     steps: str = '123',
# ):
#     """the main function to filter obj

#     Args:
#         obj (dict): obj, obj should contain only one label
#         out_dir (str): the out dir for saving filtered obj pkl files
#         prefix (str): the output pkl file prefixs
#         steps (str, optional): three steps to filter obj. Defaults to '123'.
#                                step1: filter out reads without window or with invalid window
#                                step2: filter out reads by 3d
#                                step3: filter out reads by density. the density matrix and cutoff are from step2,
#                                       while input for this step is the result of step1.
#     """
#     obj_raw = copy.deepcopy(obj)
#     all_labels = np.unique([read_obj['label'] for read_id, read_obj in obj_raw.items()])
#     assert len(all_labels) == 1
#     prefix = prefix if prefix else all_labels[0]

#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     if '1' in steps:
#         filter_out_invalid_reads_and_add_window_att(obj=obj_raw, min_len=1000)
#         io.save_pickle(obj=obj_raw, save_to_file=f'{out_dir}/{prefix}.raw.pkl')
    
#     if '2' in steps:
#         obj_3d, three_d_cutoff_dict = clean_obj(obj=obj_raw, sd_fold=1, clean_by_label=True)
#         io.save_pickle(obj=obj_3d, save_to_file=f'{out_dir}/{prefix}.3d.clean.pkl')
    
#     if '3' in steps:
#         all_labels = np.unique([read_obj['label'] for read_id, read_obj in obj_3d.items()])
#         density_matrix_dict, den_cutoff_dict = dn.get_density_matirx_and_cutoff_for_each_label_in_two_objs(obj_3d=obj_3d, obj_valid=obj_raw, labels=all_labels)


def get_valid_reads_from_objs(
    objs: List[Union[dict, str]],
    prefix: str,
    save_dir: str = None,
    split: bool = False,
    filter_by_dna1_dna2: bool = True,
    dna1_min_len: int = 8900,
    dna1_max_len: int = 12900,
    dna2_min_len: int = 7800,
    dna2_max_len: int = 12200,
) -> Union[None, Tuple[dict, pd.DataFrame], Tuple[dict, dict, pd.DataFrame]]:
    """read in objs or obj files and filter reads. And combine all filtered reads to return a dict or save as a pkl file.
    filter:
    1. filter out reads without window, opo_read_num
    2. filter out reads with window length shorter than 1k, remove_1000_num
    3. filter out reads with negative signals in window, remove_negative_num
    4. filter out reads with signals higher than I0 in window, remove_I0_num
    5. filter out read by dna1_len and dna2_len

    Args:
        objs (list[dict, str]): objs or obj files
        filter_by_dna1_dna2 (bool): whether to fliter reads by dna1_len and dna2_len. Default is True,
        prefix (str): add read with label `prefix`.
                      if `save_dir` is not None, save obj to `save_dir` with this prefix, file name would be `prefix`_valid.pkl. 
                      And save stats as `prefix`.valid.csv file. 
        save_dir (str): if not None, save obj to this dir. if `split` is True, save `prefix`_valid80.pkl, `prefix`_valid20.pkl and `prefix`_valid.csv
                        if is None, and split is False return obj and stat_df.
                        if is None, and split is True return valid80_obj, valid20_obj, stat_df
        split (bool): whether to split valid_obj into 80% and 20%.

    Returns:
        Union[None, Tuple[dict, pd.DataFrame], Tuple[dict, dict, pd.DataFrame]]: 1. if `save_dir` is None and `split` is False (default), return valid_obj, stat_df
                                                                                 2. if `save_dir` is None and `split` is True, return valid80_obj, valid20_obj, stat_df
                                                                                 3. if `save_dir` is not None, write *obj and stat_df to file
    """
    
    valid_objs, stat_dfs = [], []
    for obj in objs:
        one_valid_obj, one_stat_df = _get_valid_reads_from_one_obj(obj, filter_by_dna1_dna2=filter_by_dna1_dna2,
                                                                   dna1_min_len=dna1_min_len,
                                                                   dna1_max_len=dna1_max_len,
                                                                   dna2_min_len=dna2_min_len,
                                                                   dna2_max_len=dna2_max_len)
        valid_objs.append(one_valid_obj)
        stat_dfs.append(one_stat_df)
    stat_df = pd.concat(stat_dfs, ignore_index=True).sum().to_frame().T
    stat_df.index = [prefix]

    if isinstance(objs[0], str):
        sample_name_list = [re.search(r'(\S+)\.pkl', os.path.basename(one_file_name)).group(1) for one_file_name in objs]
    else:
        sample_name_list = [f'{prefix}_sample{i+1}' for i, one_obj in enumerate(objs)]

    valid_obj = tl.combine_objs(obj_list=valid_objs, sample_name_list=sample_name_list, label_list=[prefix]*len(valid_objs), get_obj_stat=False)

    if split:
        valid80_obj, valid20_obj = split_obj_into_train_test_obj_by_ratio(obj=valid_obj, test_ratio=0.2, seed=0)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if split:
            io.save_pickle(obj=valid80_obj, save_to_file=f'{save_dir}/{prefix}_valid80.pkl')
            io.save_pickle(obj=valid20_obj, save_to_file=f'{save_dir}/{prefix}_valid20.pkl')
        else:
            io.save_pickle(obj=valid_obj, save_to_file=f'{save_dir}/{prefix}_valid.pkl')
        stat_df.to_csv(f'{save_dir}/{prefix}_valid_stat.csv')
        return None
    if split:
        return valid80_obj, valid20_obj, stat_df
    else:
        return valid_obj, stat_df
        
        

def _get_valid_reads_from_one_obj(
    obj: Union[dict, str],
    filter_by_dna1_dna2: bool = True,
    dna1_min_len: int = 8900,
    dna1_max_len: int = 12900,
    dna2_min_len: int = 7800,
    dna2_max_len: int = 12200,
) -> (dict, pd.DataFrame):
    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    elif isinstance(obj, dict):
        obj = copy.deepcopy(obj)
    
    stat_descriptions = ['all', 'opo', 'remove_1000', 'remove_negative', 'remove_I0']
    stats = []

    stats.append(len(obj))

    # step1
    filter_out_reads_without_widows(obj=obj)
    tl.set_att_for_an_obj(obj, atts=['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length', 'pd2rd', 'signal_length', 'dna1_len', 'dna2_len'], 
                          in_place=True)
    stats.append(len(obj))

    # step2
    filter_out_reads_by_window_len(obj=obj, min_len=1000)
    stats.append(len(obj))

    # step3
    filter_out_reads_with_negative_signals_in_window(obj=obj)
    stats.append(len(obj))

    # step4
    filter_out_reads_with_curr_high_than_i0_in_window(obj)
    stats.append(len(obj))

    if filter_by_dna1_dna2:
        filter_out_reads_by_dna1_len(obj, dna1_min_len=dna1_min_len, dna1_max_len=dna1_max_len)
        stats.append(len(obj))

        filter_out_reads_by_dna2_len(obj, dna2_min_len=dna2_min_len, dna2_max_len=dna2_max_len)
        stats.append(len(obj))

        stat_descriptions.extend(['dna1', 'dna2'])

    stats.append(len(obj))
    stat_descriptions.append('valid')


    stat_df = pd.DataFrame(stats, index=stat_descriptions).T
    return obj, stat_df

    

def _three_d_filter_for_one_obj(
    obj: Union[dict, str],
    prefix: str
):
    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    elif isinstance(obj, dict):
        obj = copy.deepcopy(obj)
    stat_dict = {}
    stat_dict['valid80'] = len(obj)
    threed_obj, cutoff_dict = clean_obj(obj=obj, clean_by_label=False, direct_oprate_on_obj=True)
    cutoff_df = pd.DataFrame(cutoff_dict).T
    stat_dict['3d'] = len(threed_obj)
    stat_df = pd.DataFrame(stat_dict, index=[prefix])
    return threed_obj, cutoff_df, stat_df

def three_d_filter_for_one_obj(
    obj: Union[dict, str],
    prefix: str,
    save_dir: str = None,
) -> Union[None, Tuple[dict, pd.DataFrame, pd.DataFrame]]:
    """read in valid obj or pkl file, do 3d filter into one 3d obj.

    Args:
        objs (Union[dict, str]): obj or pkl file for valid reads
        prefix (str): if save_dir is not None, save 3d obj as `save_dir`/`prefix`_3d.pkl, cutoff_df as `save_dir`/`prefix`_3d_cutoff.csv, stat_df as `save_dir`/`prefix`_3d_stat.csv
                      `prefix` would also use as the index of stat_df
        save_dir (str, optional): the dir to save 3d pkl file, cutoff_df, stat_df. Defaults to None.

    Returns:
        Union[None, Tuple[dict, pd.DataFrame]]: if save_dir is not None, return None, else return 3d obj and cutoff_df
    """
    threed_obj, cutoff_df, stat_df = _three_d_filter_for_one_obj(obj=obj, prefix=prefix)
    if save_dir and prefix:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        io.save_pickle(obj=threed_obj, save_to_file=f'{save_dir}/{prefix}_3d.pkl')
        cutoff_df.to_csv(f'{save_dir}/{prefix}_3d_cutoff.csv')
        stat_df.to_csv(f'{save_dir}/{prefix}_3d_stat.csv')
    return threed_obj, cutoff_df, stat_df



def density_filter_for_one_obj(
    threed_obj: Union[dict, str],
    valid_obj: Union[dict, str],
    prefix: str,
    save_dir: str = None,
    # target: Literal['dna1', 'window', 'dna2'] = 'window',
) -> Union[None, Tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]]:
    """step1: get density_matrix and cutoff from threed obj
       step2: calculate llikelihood for reads in valid obj to density matrix
       step3: select high quality reads that pass the cutoff

    Args:
        threed_obj (Union[dict, str]): threed_obj
        valid_obj (Union[dict, str]): valid_obj
        prefix (str): if save_dir is not None, save density obj as `save_dir`/`prefix`_density.pkl, 
                      density_matrix_dict as `save_dir`/`prefix`_density_matrix.pkl,
                      cutoff_dict as `save_dir`/`prefix`_density_cutoff.pkl, 
                      llikelihood_df as `save_dir`/`prefix`_density_llikelihood.csv, 
                      stat_df as `save_dir`/`prefix`_desity_stat.csv.
                      `prefix` would also use as the index of stat_df
        save_dir (str, optional): the dir to save density pkl file, cutoff_dict pkl file, stat_df. Defaults to None.
        # target (Literal['dna1', 'window', 'dna2']): extract this part of the read to calculate density. Defaults to 'window'.

    Returns:
        Union[None, Tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]]: density_obj, density_matrix_dict, cutoff_dict, llikelihood_df, stat_df
    """
    threed_obj = io.read_obj_or_pickle(threed_obj)
    valid_obj = io.read_obj_or_pickle(valid_obj)
    stat_dict = {}
    stat_dict['valid80'] = len(valid_obj)
    
    # step1
    density_matrix_dict, cutoff_dict = dn.get_density_matirx_and_cutoff_for_each_label_in_two_objs(
        obj_3d=threed_obj, obj_valid=valid_obj, labels=[prefix]
        )
    
    # step2
    llikelihood_df = dn.get_llikelihood_between_reads_of_an_obj_to_density_matrix_dict(
        obj=valid_obj, density_matrix_dict=density_matrix_dict,
        )
        
    # step3
    read_ids_need_to_remove = list(llikelihood_df[llikelihood_df[prefix]<cutoff_dict[prefix]].index)
    density_obj = tl.delete_reads_in_an_obj(obj_dict=valid_obj, read_ids_to_be_removed=read_ids_need_to_remove, in_place=False)
    stat_dict['density'] = len(density_obj)
    stat_df = pd.DataFrame(stat_dict, index=[prefix])

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        io.save_pickle(obj=density_obj, save_to_file=f'{save_dir}/{prefix}_density.pkl')
        io.save_pickle(obj=density_matrix_dict, save_to_file=f'{save_dir}/{prefix}_density_matrix.pkl')
        io.save_pickle(obj=cutoff_dict, save_to_file=f'{save_dir}/{prefix}_density_cutoff.pkl')
        llikelihood_df.to_csv(f'{save_dir}/{prefix}_density_llikelihood.csv')
        stat_df.to_csv(f'{save_dir}/{prefix}_density_stat.csv')
        return None

    return density_obj, density_matrix_dict, cutoff_dict, llikelihood_df, stat_df


def filter_an_obj_by_distance_to_density_matrix_dict(
    obj: Union[dict, str],
    density_matrix_dict: dict,
    density_cutoff_df: pd.DataFrame,
    prefix: str,
    save_dir: str = None,
) -> Union[None, Tuple[dict, pd.DataFrame]]:
    """Calculate distance of all reads in an obj to denstity_matrix_dict, and select reads that satisfy one of the density matrix cutoff.

    Args:
        obj (Union[dict, str]): obj or file path
        density_matrix_dict (dict): key: label, value: density matrix
        density_cutoff_df (pd.DataFrame): cutoff for all lables, each column is a label
        prefix (str): if save_dir is not None, save density obj as `save_dir`/`prefix`.pkl, 
                      stat_df as `save_dir`/`prefix`_stat.csv.
        save_dir (str, optional): the dir to save density pkl file, stat_df

    Returns:
        Union[None, Tuple[dict, pd.DataFrame]]: density_obj, stat_df
    """
    obj = io.read_obj_or_pickle(obj)
    stat_dict = {}
    stat_dict['valid'] = len(obj)
    llikelihood_df = dn.get_llikelihood_between_reads_of_an_obj_to_density_matrix_dict(obj=obj, density_matrix_dict=density_matrix_dict)
    density_cutoff_df = density_cutoff_df[llikelihood_df.columns[0:-1]]
    read_ids_need_to_removed = llikelihood_df[np.sum((llikelihood_df.iloc[:,0:-1] - density_cutoff_df.iloc[0])>0, axis=1)==0].index
    density_obj = tl.delete_reads_in_an_obj(obj_dict=obj, read_ids_to_be_removed=read_ids_need_to_removed, in_place=False)
    stat_dict['mix_density'] = len(density_obj)
    stat_df = pd.DataFrame(stat_dict, index=[prefix])

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        io.save_pickle(obj=density_obj, save_to_file=f'{save_dir}/{prefix}.pkl')
        stat_df.to_csv(f'{save_dir}/{prefix}_stat.csv')
        return None
    return density_obj, stat_df


def threed_mix_filter_for_an_valid_obj(
    obj: Union[dict, str],
    threed_cutoffs_dict: dict,
    prefix: str,
    save_dir: str = None,
) -> Union[None, Tuple[dict, pd.DataFrame]]:
    """do threed mix filter for an obj. a read meets all three cutoffs for any pep would be selectd.

    Args:
        obj (Union[dict, str]): obj
        threed_cutoffs_dict (dict): key: pep_name; value: a pd.Series of cutoffs
        prefix (str): if save_dir is not None, save density obj as `save_dir`/`prefix`_3d.pkl, 
                      stat_df as `save_dir`/`prefix`_3d_stat.csv.
        save_dir (str, optional): the dir to save pkl file, stat_df

    Returns:
        Union[None, Tuple[dict, pd.DataFrame]]: 3d.pkl, stat_df
    """
    obj = io.read_obj_or_pickle(obj=obj)
    read_ids_need = []
    for read_id, read_obj in obj.items():
        for pep_name, cutoff_s in threed_cutoffs_dict.items():
            if is_a_read_obj_meet_threed(read_obj, cutoff_s):       
                read_ids_need.append(read_id)
                break
    new_obj = extract_reads_as_an_obj(obj=obj, read_ids=read_ids_need)
    
    stat_dict = {}
    stat_dict['valid'] = len(obj)
    stat_dict['mix_3d'] = len(new_obj)
    stat_df = pd.DataFrame(stat_dict, index=[prefix])

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        io.save_pickle(obj=new_obj, save_to_file=f'{save_dir}/{prefix}_3d.pkl')
        stat_df.to_csv(f'{save_dir}/{prefix}_3d_stat.csv')
        return None
    return new_obj, stat_df

        


def is_a_read_obj_meet_threed(
    read_obj: dict,
    cutoff_s: pd.Series,
):
    meet_pd2rd = 1 if cutoff_s['pd2rd_left'] < read_obj['pd2rd'] < cutoff_s['pd2rd_right'] else 0
    meet_mean = 1 if cutoff_s['window_i2i0_mean_left'] < read_obj['window_i2i0_mean'] < cutoff_s['window_i2i0_mean_right'] else 0
    meet_std = 1 if cutoff_s['window_i2i0_std_left'] < read_obj['window_i2i0_std'] < cutoff_s['window_i2i0_std_right'] else 0
    if meet_pd2rd + meet_mean + meet_std == 3:
        return True
    return False        



