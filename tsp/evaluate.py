#!/usr/bin/env python
import numpy as np
import pickle
from itertools import tee
import os
import random

DIR = os.path.dirname(__file__)
coords = pickle.load(open(os.path.join(DIR, "bier127.dat"), "rb"))

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def eucdist(src, dst):
    _src = np.array(coords[src])
    _dst = np.array(coords[dst])
    return np.sqrt(np.sum((_src - _dst) ** 2))

def evaluate(answer_list):
    answer = np.array(answer_list)
    if (answer.shape != (127,)):
        raise Exception('Input data is invalid: there aren\'t 127 indices.')
    if (np.isnan(answer).any()):
        raise Exception('Input data is invalid: Non-numerial data exists in dataset.')
    if (sorted(list(answer)) != list(range(127))):
        raise Exception('Input is not a permutation of integers from 0 to 126.')

    travels = pairwise(list(answer) + [list(answer)[0]])
    dist = 0.0
    for travel in travels:
        src, dst = travel
        dist += eucdist(src, dst)
    return dist

if __name__ == "__main__":
    p = list(range(127))
    random.shuffle(p)
    print(evaluate(p))
    random.shuffle(p)
    print(evaluate(p))
    random.shuffle(p)
    print(evaluate(p))

