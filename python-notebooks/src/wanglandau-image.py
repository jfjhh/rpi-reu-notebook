#!/usr/bin/env python
# coding: utf-8

# ## Thermal calculations on images

from numba import njit
from numba.experimental import jitclass
from numba import int64


import numpy as np
from scipy import interpolate, special
import os
import time
import tempfile
import h5py, hickle
from multiprocessing import Pool
from pprint import pprint


# We extend the path instead of using `src.module` to be able to run generated files.
import sys
if 'src' not in sys.path: sys.path.append('src')
import wanglandau as wl


integer = int64
spec = [
    ('I0', integer[:]),
    ('I',  integer[:]),
    ('N',  integer),
    ('M',  integer),
#     ('Es', integer[:]),
    ('E',  integer),
    ('Eν', integer),
    ('dE', integer),
    ('dx', integer),
    ('i',  integer)
]

@jitclass(spec)
class StatisticalImage:
    def __init__(self, I0, I, M):
        if len(I0) != len(I):
            raise ValueError('Ground image I0 and current image I should have the same length.')
        if M < 0:
            raise ValueError('Maximum site value must be nonnegative.')
        self.I0 = I0
        self.I  = I
        self.N  = len(I0)
        self.M  = M
#         self.Es = self.energy_bins()
        self.E  = self.energy()
        self.Eν = self.E
        self.dE = 0
        self.dx = 0
        self.i  = 0
    def state(self):
        return self.I0.copy(), self.I.copy(), self.M
    def state_names(self):
        return 'I0', 'I', 'M'
    def copy(self):
        return StatisticalImage(*self.state())
    def energy_bins(self):
        E0 = 0
        Ef = np.sum(np.maximum(self.I0, self.M - self.I0))
        ΔE = 1
        return np.arange(E0, Ef + ΔE + 1, ΔE)
    def energy(self):
        return np.sum(np.abs(self.I - self.I0))
    def propose(self):
        i = np.random.randint(self.N)
        self.i = i
        x0 = self.I0[i]
        x = self.I[i]
        r = np.random.randint(2)
        if x == 0:
            dx = r
        elif x == self.M:
            dx = -r
        else:
            dx = 2*r - 1
        dE = np.abs(dx) if x0 == x else (dx if x0 < x else -dx)
        self.dx = dx
        self.dE = dE
        self.Eν = self.E + dE
    def accept(self):
        self.I[self.i] += self.dx
        self.E = self.Eν


# ### Parallel Simulation

N = 16
Moff = 0
I0 = Moff * np.ones(N, dtype=int)
system_parameters = {
    'I0': I0,
    'I': I0.copy(),
    'M': 2**5 - 1
}
parallel_parameters = {
    'bins': 8,
    'overlap': 0.5,
    'steps': 1_000_000
}
wl_parameters = {
    'M': 1_000_000,
    'ε': 1e-10,
    'logf0': 1,
    'flatness': 0.1,
    'logging': False
}

system = StatisticalImage(**system_parameters) # Intermediate value
Es = system.energy_bins()


print('Parallel Wang-Landau simulation with')
for k, v in wl_parameters.items():
    print("\t", k, '\t', v)
print('on a {}.'.format(system.__class__.__name__))


def parallel_wanglandau(subsystem): # Convenient form for `Pool.map`
    wl.urandom_reseed()
    state, Es = subsystem
    system = StatisticalImage(*state)
    print('(', end='', flush=True)
#     results = wl.wanglandau(system, Es, M = wlM, ε = 1e-8, logging=False)
    results = wl.wanglandau(system, Es, **wl_parameters)
    print(')', end='', flush=True)
    return results


print('Finding parallel bin systems ... ', end='', flush=True)
psystems = wl.parallel_systems(system, Es, **parallel_parameters)
print('done.')


print('Running | ', end='', flush=True)
start_time = time.time()
with Pool() as pool:
    wlresults = pool.map(parallel_wanglandau, psystems)
print(' | done in', int(time.time() - start_time), 'seconds.')


sEs, sS = wl.stitch_results(wlresults)


with tempfile.NamedTemporaryFile(mode='wb', prefix='wlresults-image-', suffix='.hdf5', dir='data', delete=False) as f:
    with h5py.File(f, 'w') as hkl:
        print('Writing results ... ', end='', flush=True)
        d = {}
        for k, v in zip(system.state_names(), system.state()):
            d.update({k: v})
        hickle.dump({
            'parameters': {
                'system': d,
                'wanglandau': {},
                'parallel': {}
            },
            'results': {
                'Es': sEs,
                'S': sS
            },
            'parallel_results': wlresults
        }, hkl)
        print('done: {}'.format(os.path.relpath(f.name)))


# ### Results

import matplotlib.pyplot as plt


N, M = len(system_parameters['I0']), system_parameters['M']


for Es, S, H in wlresults:
    plt.plot(Es[:-1], S)


wlEs, S = sEs[:-1], sS


# Fit a spline to interpolate and optionally clean up noise, giving WL g's up to a normalization constant.

gspl = interpolate.splrep(wlEs, S, s=0*np.sqrt(2))
wlgs = np.exp(interpolate.splev(wlEs, gspl) - min(S))


# ### Exact solution

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
plt.plot(gray_Es / len(gray_Es), np.log(gray_gs), 'gray', label='Gray')
plt.plot(wlEs / len(wlEs), np.log(wlgs), label='WL')
plt.xlabel('E / MN')
plt.ylabel('ln g')
plt.title('N = {}, M = {}'.format(N, M))
plt.legend();


plt.plot(wlEs / len(wlEs), np.abs(wlgs - bw_gs) / bw_gs)
plt.title('Relative error')
plt.xlabel('E / MN')
plt.ylabel('ε(S)');


print('End of job.')

