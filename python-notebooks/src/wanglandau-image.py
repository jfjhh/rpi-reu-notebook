#!/usr/bin/env python
# coding: utf-8

# ## Thermal calculations on images

import numpy as np
from scipy import interpolate, special


import sys
if 'src' not in sys.path: sys.path.append('src')
import wanglandau as wl


# ### Parallel Simulation

N = 16
M = 2**5 - 1
I0 = np.zeros(N, dtype=int)
system_params = {
    'StatisticalImage': {
        'I0': I0,
        'I': I0.copy(),
        'M': M
    }
}


# L = 16
# system_params = {
#     'IsingModel': {
#         'spins': np.ones((L, L), dtype=int)
#     }
# }


params = {
    'system': system_params,
    'simulation': {
        'max_sweeps': 500_000_000,
        'flat_sweeps': 10_000,
        'eps': 1e-8,
        'logf0': 1,
        'flatness': 0.1
    },
    'parallel': {
        'bins': 8,
        'overlap': 0.25,
        'sweeps': 1_000_000
    },
    'save': {
        'prefix': 'simulation-',
        'dir': 'data'
    }
}


params.pop('parallel', None) # Single run
wlresults = wl.run(params, log=True)


[r['converged'] for r in wlresults['results']]


# ### Results

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12


import h5py, hickle
with h5py.File('data/simulation-gozi5xqv.h5', 'r') as f:
    wlresults = hickle.load(f)
system_params = wlresults['parameters']['system']


wlEs, S, ΔS = wl.join_results([wlresults['results']])


for i, r in enumerate([wlresults['results']]):
    plt.plot(r['Es'][:-1], r['S'] + ΔS[i])


N, M = len(system_params['StatisticalImage']['I0']), system_params['StatisticalImage']['M']


# Fit a spline to interpolate and optionally clean up noise, giving WL g's up to a normalization constant.

gspl = interpolate.splrep(wlEs, S, s=0*np.sqrt(2))
wlgs = np.exp(interpolate.splev(wlEs, gspl) - min(S))


# ### Exact density of states
# 
# We only compute to halfway since $g$ is symmetric and the other half's large numbers cause numerical instability.

def reflect(a, center=True):
    if center:
        return np.hstack([a[:-1], a[-1], a[-2::-1]])
    else:
        return np.hstack([a, a[::-1]])


# The exact density of states for uniform values. This covers the all gray and all black/white cases. Everything else (normal images) are somewhere between. The gray is a slight approximation: the ground level is not degenerate, but we say it has degeneracy 2 like all the other sites. For the numbers of sites and values we are using, this is insignificant.

def bw_g(E, N, M, exact=True):
    return sum((-1)**k * special.comb(N, k, exact=exact) * special.comb(E + N - 1 - k*(M + 1), E - k*(M + 1), exact=exact)
        for k in range(int(E / M) + 1))
def exact_bw_gs(N, M):
    Es = np.arange(N*M + 1)
    gs = np.vectorize(bw_g)(np.arange(1 + N*M // 2), N, M, exact=False)
    return Es, reflect(gs, len(Es) % 2 == 1)


def gray_g(E, N, M, exact=True):
    return 2 * bw_g(E, N, M, exact=exact)
def exact_gray_gs(N, M):
    Es = np.arange(N*M + 1)
    gs = np.vectorize(gray_g)(np.arange(1 + N*M // 2), N, M, exact=False)
    return Es, reflect(gs, len(Es) % 2 == 1)


# Expected results for black/white and gray.

bw_Es, bw_gs = exact_bw_gs(N=N, M=M)
gray_Es, gray_gs = exact_gray_gs(N=N, M=-1 + (M + 1) // 2)


# Choose what to compare to.

Es, gs = bw_Es, bw_gs


# Presumably all of the densities of states for different images fall in the region between the all-gray and all-black/white curves.

plt.plot(bw_Es / len(bw_Es), np.log(bw_gs), 'black', label='BW')
plt.plot(wlEs / len(wlEs), S - min(S), '#ff6716', label='WL')
plt.xlabel('E / MN')
plt.ylabel('ln g')
plt.title('N = {}, M = {}'.format(N, M))
plt.legend()
plt.savefig('wanglandau-bw.png', dpi=600)


# plt.plot(wlEs / len(wlEs), np.abs(wlgs - bw_gs) / bw_gs)
# plt.ylabel('Relative error')
plt.plot(wlEs / len(wlEs), S - np.log(bw_gs) - min(S))
plt.ylabel('Residuals')
plt.xlabel('E / MN');


print('End of job.')

