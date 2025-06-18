import pickle
import numpy as np
import pandas as pd
import copy
import os
import re
import glob
import configparser
from configparser import ConfigParser, ExtendedInterpolation
from typing import Dict, Optional, Tuple, Union, Literal, List

def get_read_number(
    obj: dict,
    targets: list = ['total', 'read_with_window'],
    return_df_or_dict: Literal['df', 'dict'] = 'dict',
    sample_name: str = 'sample',
    save: bool = False,
    save_dir: str = './',
    save_file_name: str = 'sample.csv',   
) -> Union[dict, pd.DataFrame]:
    read_num_dict = {}
    for target in targets:
        read_num_dict[target] = 0
    for read_id, read_obj in obj.items():
        if 'total' in targets:
            read_num_dict['total'] += 1
        if 'read_with_window' in targets:
            if 'window' in read_obj and read_obj['window'] != None:
                read_num_dict['read_with_window'] += 1
    if return_df_or_dict == 'df':
        res = pd.DataFrame(read_num_dict, index=[sample_name])
    else:
        res = read_num_dict

    if save:
        assert return_df_or_dict == 'df'
        res.to_csv(os.path.join(save_dir, save_file_name))
    else:
        return res