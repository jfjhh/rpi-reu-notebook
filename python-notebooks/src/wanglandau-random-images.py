#!/usr/bin/env python
# coding: utf-8

# ## Comparison of Wang-Landau results for random Statistical Images

import numpy as np
from scipy import interpolate, special
import os, h5py, hickle
import matplotlib.pyplot as plt


import sys
if 'src' not in sys.path: sys.path.append('src')
import wanglandau as wl


datadir = 'data/random-images'
paths = [os.path.join(datadir, f) for f in os.listdir(datadir)]


def file_results(path):
    with h5py.File(path, 'r') as f:
        result = hickle.load(f)
        Es = result['results']['Es'][:-1]
        S = result['results']['S']
        return Es, S - min(S)


for result in map(file_results, paths):
    plt.plot(*result, 'black', alpha=0.25, linewidth=0.25)

