#!/usr/bin/env python
# coding: utf-8

# # The Wang-Landau algorithm (density of states)
# We determine thermodynamic quantities from the partition function by obtaining the density of states from a simulation.

from numba import njit
import numpy as np


import sys
if 'src' not in sys.path: sys.path.append('src')
import simulation as sim


# Utility functions for the simulation.

@njit(cache=True, inline='always')
def bisect_right(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

@njit(cache=True)
def binindex(a, x):
    return bisect_right(a, x, lo=0, hi=len(a) - 1) - 1


@njit(cache=True)
def flat(H, ε = 0.2):
    """Determines if a histogram is approximately flat to within ε of the mean height."""
    return not np.any(H < (1 - ε) * np.mean(H)) and np.all(H != 0)


# ## Algorithm

# A Wang-Landau algorithm, with quantities as logarithms and with monte-carlo steps proportional to $f^{-1/2}$ (a "Zhou-Bhat schedule").
# 
# We use energy bins encoded by numbers $E_i$ for $i \in [0,\, N]$, so that there are $N$ bins. The energies $E$ covered by bin $i$ satisfy $E_i \le E < E_{i+1}$. For the bounded discrete systems that we are considering, we must choose $E_N$ to be an arbitrary number above the maximum energy.

@njit(cache=True)
def simulation(system,
                Es,             # The energy bins
                M = 1_00_000,   # Monte carlo step scale
                ε = 1e-10,      # f tolerance
                logf0 = 1,      # Initial log f
                flatness = 0.1, # Desired histogram flatness
                logging = False # Log progress of f-steps
               ):
    if M <= 0 or ε <= 1e-16 or not (0 < logf0 <= 1) or not (0 <= flatness < 1):
        raise ValueError('Invalid Wang-Landau parameter.')

    # Initial values
    E0 = Es[0]
    Ef = Es[-1]
    N = len(Es) - 1
    logf = 2 * logf0
    logftol = np.log(1 + ε)
    S = np.zeros(N) # Set all initial g's to 1
    H = np.zeros(N, dtype=np.int32)
    i = binindex(Es, system.E)
    converged = True
    
    if logging:
        mciters = 0
        fiter = 0
        fiters = int(np.ceil(np.log2(logf0) - np.log2(logftol)))
        print("Wang-Landau START")
    
    while logftol < logf:
        H[:] = 0
        logf /= 2
        iters = 0
        niters = int((M + 1) * np.exp(-logf / 2))
        if logging:
            fiter += 1
        while not flat(H, flatness) and iters < niters:
            system.propose()
            Eν = system.Eν
            j = binindex(Es, Eν)
            if E0 <= Eν < Ef and (
                S[j] < S[i] or np.random.rand() < np.exp(S[i] - S[j])):
                system.accept()
                i = j
            H[i] += 1
            S[i] += logf
            iters += 1
        mciters += iters
        if niters <= iters:
            converged = False
        if logging:
            print("f: ", fiter, " / ", fiters, "\t(", iters, " / ", niters, ")")
    
    if logging:
        print("Done: ", mciters, " total MC iterations.")
    return Es, S, H


# ### Parallel construction of the density of states

@njit
def find_bin_systems(system, Es, Ebins, N = 1_000_000, method = 'wl'):
    """
    Find systems with energies in the bins given by `Es` by stepping `sys`.
    
    Args:
        system: The initial system to search from. This is usually a ground state.
        Es: The energies of the system.
        Ebins: The energy bins to find systems for.
        N: The maximum number of steps to try.
        method: The string name of the search method to try.
            'wl': Wang-Landau steps where we prefer energies we have not visited
            'increasing': Only accept increases in energy. This only works for
                steps that are not trapped by local maxima of energy.
    
    Returns:
        A list of independent systems with energies in Ebins.
    
    Raises:
        ValueError: The method argument was invalid.
        RuntimeError: Bin systems could not be found after N steps.
    """
    if method == 'wl':
        S = np.zeros(len(Es), dtype=np.int32)
    systems = [None] * (len(Ebins) - 1)
    n = 0
    l = len(Ebins) - 1
    systems = [system] * l
    empty = np.repeat(True, l)
    i = binindex(Es, system.E)
    while np.any(empty) and n < N:
        for s in range(l):
            if empty[s] and Ebins[s] <= system.E < Ebins[s + 1]:
                systems[s] = system.copy()
                empty[s] = False
        
        system.propose()
        j = binindex(Es, system.Eν)
        if method == 'wl':
            if S[j] < S[i]:
                i = j
                system.accept()
            S[i] += 1
        elif method == 'increasing':
            if system.E < system.Eν:
                system.accept()
        else:
            raise ValueError('Invalid method argument for finding bin systems.')
        n += 1
        
    if N <= n:
        raise RuntimeError('Could not find bin systems (hit step limit).')
    return systems


def parallel_systems(system, bins = 8, overlap = 0.1, steps = 1_000_000, method = 'wl', **kwargs):
    Es = system.energy_bins() # Intrinsic to the system
    Ebins = np.linspace(Es[0], Es[-1], bins + 1) # For parallel subsystems
    systems = find_bin_systems(system, Es, Ebins, steps, method)
    binEs = [(lambda E0, Ef: Es[(E0 <= Es) & (Es <= Ef)])(*sim.extend_bin(Ebins, i, overlap))
             for i in range(len(Ebins) - 1)]
    return zip(systems, binEs)

