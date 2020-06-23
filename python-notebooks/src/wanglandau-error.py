#!/usr/bin/env python
# coding: utf-8

# ## Simulation error of Wang-Landau results

import numpy as np
from scipy import interpolate, special
import os, h5py, hickle
import matplotlib.pyplot as plt


import sys
if 'src' not in sys.path: sys.path.append('src')
import wanglandau as wl


paths = [os.path.join('data/ising-ajp', f) for f in os.listdir('data/ising-ajp')]


with h5py.File(paths[0], 'r') as f:
    result = hickle.load(f)
    Es = result['results']['Es'][:-1]


def file_lngs(path):
    with h5py.File(path, 'r') as f:
        result = hickle.load(f)
        S = result['results']['S']
        return S - min(S)


mean_lng = np.zeros(len(Es))
std_lng = np.zeros(len(Es))
for lng in map(file_lngs, paths):
    mean_lng += lng
mean_lng /= len(paths)
for lng in map(file_lngs, paths):
    std_lng += (mean_lng - lng)**2
std_lng = np.sqrt(std_lng / (len(paths) - 1))


plt.plot(Es, mean_lng);


for lng in map(file_lngs, paths):
    plt.plot(Es, lng, alpha=0.05, linewidth=0.1)


for lng in map(file_lngs, paths):
    plt.plot(Es, lng - mean_lng, alpha=1, linewidth=0.5)


plt.plot(Es, std_lng);

