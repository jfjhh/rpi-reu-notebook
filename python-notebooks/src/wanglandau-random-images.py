#!/usr/bin/env python
# coding: utf-8

# ## Comparison of Wang-Landau results for random Statistical Images

import numpy as np
from scipy import interpolate, special
import os, h5py, hickle
import matplotlib.pyplot as plt
import pprint


import sys
if 'src' not in sys.path: sys.path.append('src')
import wanglandau as wl
from statistical_image import exact_bw_gs
import canonical_ensemble as canonical
from intensity_entropy import intensity_entropy


datadir = 'data/random-images'
paths = [os.path.join(datadir, f) for f in os.listdir(datadir)]
len(paths)


with h5py.File(paths[0], 'r') as f:
    result = hickle.load(f)
    imp = result['parameters']['system']['StatisticalImage']
    N = len(imp['I0'])
    M = imp['M']
    Es = result['results']['Es'][:-1]


pprint.pprint(result['parameters'])


def file_results(path):
    with h5py.File(path, 'r') as f:
        result = hickle.load(f)
        Es = result['results']['Es'][:-1]
        S = result['results']['S']
        return Es, S - min(S)


xEs, xgs = exact_bw_gs(N, M)
xlng = np.log(xgs)
xens = canonical.Ensemble(xEs, xlng, 'Exact')


for Es, S in map(file_results, paths):
    plt.plot(Es, S, 'black', alpha=0.05)
plt.plot(xEs, xlng);


βs = np.exp(np.linspace(-7, 4, 500))


for Es, S in map(file_results, paths):
    ens = canonical.Ensemble(Es, S)
    plt.plot(-np.log(βs), ens.entropy(βs), 'black', alpha=0.05)
plt.plot(-np.log(βs), xens.entropy(βs), 'orange');


for Es, S in map(file_results, paths):
    ens = canonical.Ensemble(Es, S)
    plt.plot(-np.log(βs), ens.entropy(βs) - xens.entropy(βs), 'black', alpha=0.05)


# Is the canonical entropy related to the intensity entropy?

result['parameters']['system']['StatisticalImage']['I0']


plt.imshow(np.reshape(result['parameters']['system']['StatisticalImage']['I0'], (int(np.sqrt(N)), -1)), cmap='gray', vmin=0, vmax=M);


plt.imshow(np.reshape(np.sort(result['parameters']['system']['StatisticalImage']['I0']), (1, -1)), cmap='gray', vmin=0, vmax=M);


I0 = result['parameters']['system']['StatisticalImage']['I0']


intensity_entropy.intensity_entropy(I0, upper=M+1)

