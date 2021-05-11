"""
Original source can be found here -> https://github.com/migalkin/StarE/blob/master/utils/utils.py
"""

import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import numpy.random as npr
from collections import namedtuple, defaultdict
from typing import Optional, List, Union, Dict, Callable, Tuple


Quint = namedtuple('Quint', 's p o qp qe')

KNOWN_DATASETS = ['wd50k', 'wikipeople', 'wd50k_100', 'wd50k_33', 'wd50k_66', 'jf17k']
RAW_DATA_DIR = Path('./data/raw_data')
PARSED_DATA_DIR = Path('./data/parsed_data')


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



def save_model(model_obj):
    """
    """
    torch.save({
        "model_state_dict": model_obj.state_dict(),
        "emb_type": model_obj.emb_type,
        "dim": model_obj.emb_dim
    }, 
        f"Hyper_kg_{model_obj.emb_type}_{model_obj.emb_dim}.tar"
    )



def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def combine(*args: Union[np.ndarray, list, tuple]):
    """
        Used to semi-intelligently combine data splits

        Case A)
            args is a single element, an ndarray. Return as is.
        Case B)
            args are multiple ndarray. Numpy concat them.
        Case C)
            args is a single dict. Return as is.
        Case D)
            args is multiple dicts. Concat individual elements

    :param args: (see above)
    :return: A nd array or a dict
    """

    # Case A, C
    if len(args) == 1 and type(args[0]) is not dict:
        return np.array(args[0])

    if len(args) == 1 and type(args) is dict:
        return args

    # Case B
    if type(args) is tuple and (type(args[0]) is np.ndarray or type(args[0]) is list):
        # Expected shape will be a x n, b x n. Simple concat will do.
        return np.concatenate(args)

    # Case D
    if type(args) is tuple and type(args[0]) is dict:
        keys = args[0].keys()
        combined = {}
        for k in keys:
            combined[k] = np.concatenate([arg[k] for arg in args], dim=-1)
        return combined

