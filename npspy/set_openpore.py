import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Literal, List
import copy

def set_openpore_for_one_signal(
    signal: Union[np.array, list],
    cutoff: float = 5.0,
) -> Union[Tuple[float, int], Tuple[None, None]]:
    """calculate openpore for a signal
    1. 找到起始端平台区域（ leading platform）
    1.1 Signal每个点原来的信号称做原始信号。
    1.2 从signal第一个点开始，每个点的信号值变成它和它之前所有点的均值，称做均值信号。
    1.3 从第一个点开始遍历，如果点x的均值信号 – 点x+1的原始信号 >= cutoff，则点x及之前的点即为leading platform。

    2. leading platform的median当做I0


    Args:
        signal (Union[np.array, list]): one signal
        cutoff (float, optional): the cutoff. Defaults to 5.0.

    Returns:
        Union[Tuple[float, int], Tuple[None, None]]: either (openpore, index) or (None, None)
    """
    running_averages = calculate_running_averages(signal)
    for i in range(len(running_averages)-1):
        if running_averages[i] - signal[i+1] >= cutoff:
            break
    if i < len(signal) - 2:
        openpore = np.median(signal[0:i+1])
    else:
        openpore = None
        i = None

    return openpore, i


def set_openpore_for_an_obj(
    obj: dict,
    cutoff: float = 5.0,
    in_palce: bool = False
) -> Union[dict, None]:
    if in_palce:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    for read_id, read_obj in obj.items():
        new_obj[read_id]['OpenPore'], new_obj[read_id]['leading_platform_index'] = set_openpore_for_one_signal(read_obj['signal'], cutoff=cutoff)
    
    if in_palce:
        return None
    return new_obj




def calculate_running_averages(numbers):
    running_totals, running_averages = np.zeros(len(numbers)), np.zeros(len(numbers))
    running_totals[0] = numbers[0]

    for i in range(1, len(numbers)):
        running_totals[i] = running_totals[i-1] + numbers[i]
    
    for i in range(len(running_totals)):
        running_averages[i] = running_totals[i] / (i+1)

    return running_averages

