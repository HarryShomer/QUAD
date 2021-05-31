"""
Original source can be found here -> https://github.com/migalkin/StarE/blob/master/utils/utils_mytorch.py
"""

from typing import List, Dict, Union
from pathlib import Path
from collections import namedtuple
import warnings
import os
import time
import json
import torch
import numpy as np
import pickle
import traceback



class ImproperCMDArguments(Exception): pass
class MismatchedDataError(Exception): pass
class BadParameters(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "Unexpected value of parameter {0}".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments

tosave = namedtuple('ObjectsToSave','fname obj')

class FancyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

class Timer:
    """ Simple block which can be called as a context, to know the time of a block. """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def default_eval(y_pred, y_true):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())

def compute_mask(t: Union[torch.Tensor, np.array], padding_idx=0):
    """
    compute mask on given tensor t
    :param t: either a tensor or a nparry
    :param padding_idx: the ID used to represented padded data
    :return: a mask of the same shape as t
    """
    if type(t) is np.ndarray:
        mask = np.not_equal(t, padding_idx)*1.0
    else:
        mask = torch.ne(t, padding_idx).float()
    return mask

# Transparent, and simple argument parsing FTW!
def convert_nicely(arg, possible_types=(bool, float, int, str)):
    """ Try and see what sticks. Possible types can be changed. """
    for data_type in possible_types:
        try:

            if data_type is bool:
                # Hard code this shit
                if arg in ['T', 'True', 'true']: return True
                if arg in ['F', 'False', 'false']: return False
                raise ValueError
            else:
                proper_arg = data_type(arg)
                return proper_arg
        except ValueError:
            continue
    # Here, i.e. no data type really stuck
    warnings.warn(f"None of the possible datatypes matched for {arg}. Returning as-is")
    return arg



class SimplestSampler:
    """
        Given X and Y matrices (or lists of lists),
            it returns a batch worth of stuff upon __next__
    :return:
    """

    def __init__(self, data, bs: int = 64):

        try:
            assert len(data["x"]) == len(data["y"])
        except AssertionError:

            raise MismatchedDataError(f"Length of x is {len(data['x'])} while of y is {len(data['y'])}")

        self.x = data["x"]
        self.y = data["y"]
        self.n = len(self.x)
        self.bs = bs  # Batch Size

    def __len__(self):
        return self.n // self.bs - (1 if self.n % self.bs else 0)

    def __iter__(self):
        self.i, self.iter = 0, 0
        return self

    def __next__(self):
        if self.i + self.bs >= self.n:
            raise StopIteration

        _x, _y = self.x[self.i:self.i + self.bs], self.y[self.i:self.i + self.bs]
        self.i += self.bs

        return _x, _y


