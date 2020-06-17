#!/usr/bin/env python
# coding: utf-8

# # The Wang-Landau algorithm (density of states)
# We determine thermodynamic quantities from the partition function by obtaining the density of states from a simulation.

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, special


# A Wang-Landau algorithm, with quantities as logarithms and with monte-carlo steps proportional to $f^{-1/2}$ (a "Zhou-Bhat schedule").

def flat(H, tol = 0.2):
    """Determines if an evenly-spaced histogram is approximately flat."""
    Hμ = np.mean(H)
    Hf = np.max(H)
    H0 = np.min(H)
    return Hf / (1 + tol) < Hμ < H0 / (1 - tol)
# def flat(H, tol = 0.2):
#     """Determines if an evenly-spaced histogram is approximately flat."""
#     Hμ = np.mean(H)
#     return not np.any(H < (1 - tol) * Hμ) and np.all(H != 0)


def wanglandau(system,
                M = 1_00_000,  # Monte carlo step scale
                ε = 1e-8,       # f tolerance
                logf0 = 1,      # Initial log f
                N = 8**2 + 1,   # Number of energy bins
                E0 = -2 * 8**2, # Minimum energy
                Ef = 2 * 8**2   # Maximum energy
               ):
    # Initial values
    logf = logf0
    logftol = np.log(1 + ε)
    Es = np.linspace(E0, Ef, N)
    S = np.zeros(N) # Set all initial g's to 1
    H = np.zeros(N, dtype=int)
    # Linearly bin the energy
    i = max(0, min(N - 1, int(round((N - 1) * (system.E - E0) / (Ef - E0)))))
    
    # Logging
    mciters = 0
    fiter = 0
    ΔE = (Ef - E0) / (N - 1)
    fiters = int(np.ceil(np.log2(logf0) - np.log2(logftol)))
    print("ΔE = {}".format(ΔE))
    
    while logftol < logf:
        H[:] = 0
        logf /= 2
        iters = 0
        niters = int((M + 1) * np.exp(-logf / 2))
        fiter += 1
        while not flat(H[2:-2]) and iters < niters: # Ising-specific histogram
#         while not flat(H) and iters < niters:
            system.propose()
            Eν = system.Eν
            j = max(0, min(N - 1, int(round((N - 1) * (Eν - E0) / (Ef - E0)))))
            if E0 - ΔE/2 <= Eν <= Ef + ΔE/2 and (S[j] < S[i] or np.random.rand() < np.exp(S[i] - S[j])):
                system.accept()
                i = j
            H[i] += 1
            S[i] += logf
            iters += 1
        mciters += iters
        print("f: {} / {}\t({} / {})".format(fiter, fiters, iters, niters))
    
    print("Done: {} total MC iterations.".format(mciters))
    return Es, S, H


# ## The 2D Ising model

class Ising:
    def __init__(self, n):
        self.n = n
        self.spins = np.sign(np.random.rand(n, n) - 0.5)
        self.E = self.energy()
        self.Eν = self.E
    def neighbors(self, i, j):
        return np.hstack([self.spins[:,j].take([i-1,i+1], mode='wrap'),
                          self.spins[i,:].take([j-1,j+1], mode='wrap')])
    def energy(self):
        return -0.5 * sum(np.sum(s * self.neighbors(i, j))
                         for (i, j), s in np.ndenumerate(self.spins))
    def propose(self):
        i, j = np.random.randint(self.n), np.random.randint(self.n)
        self.i, self.j = i, j
        dE = 2 * np.sum(self.spins[i, j] * self.neighbors(i, j))
        self.dE = dE
        self.Eν = self.E + dE
    def accept(self):
        self.spins[self.i, self.j] *= -1
        self.E = self.Eν


# Note that this class-based approach adds some overhead. For speed, instances of `Ising` should be inlined into the simulation method.

isingn = 8
sys = Ising(isingn)
Es, S, H = wanglandau(sys);


# The energies at indices 1 and -1 are not occupied in the Ising model.

Es, S, H = Es[2:-2], S[2:-2], H[2:-2]


plt.plot(Es / isingn**2, S)
plt.xlabel("E / N")
plt.ylabel("log g(E) + C");


plt.plot(Es / isingn**2, H)
plt.xlabel("E / N")
plt.ylabel("Visits");


# ### Calculating canonical ensemble averages

gspl = interpolate.splrep(Es, S, s=2*np.sqrt(2))
gs = np.exp(interpolate.splev(Es, gspl) - min(S))


plt.plot(Es / isingn**2, S)
plt.plot(Es / isingn**2, interpolate.splev(Es, gspl))
plt.xlabel("E / N")
plt.ylabel("log g(E) + C");


# Translate energies to have minimum zero so that $Z$ is representable.

nEs = Es - min(Es)


Z = lambda β: np.sum(gs * np.exp(-β * nEs))


# Ensemble averages

βs = [np.exp(k) for k in np.linspace(-3, 1, 200)]
Eμ = lambda β: np.sum(nEs * gs * np.exp(-β * nEs)) / Z(β)
E2 = lambda β: np.sum(nEs**2 * gs * np.exp(-β * nEs)) / Z(β)
CV = lambda β: (E2(β) - Eμ(β)**2) * β**2
F  = lambda β: -np.log(Z(β)) / β
Sc = lambda β: β*Eμ(β) + np.log(Z(β))


# Heat capacity

plt.plot(np.log(βs), [CV(β) for β in βs])
plt.xlabel("ln β")
plt.ylabel("Heat capacity")
plt.show()


# Entropy

plt.plot(np.log(βs), [Sc(β) for β in βs])
plt.xlabel("ln β")
plt.ylabel("S(β) + C")
plt.show()


# ## Thermal calculations on images

class StatisticalImage:
    def __init__(self, I0):
        self.I0 = I0
        self.I = I0.copy()
        self.w, self.h = np.shape(I0)
        self.E = self.energy()
        self.Eν = self.E
    def energy(self):
        return sum(x0 - x if x < x0 else x - x0
                   for x, x0 in zip(self.I.flat, self.I0.flat))
    def propose(self):
        i, j = np.random.randint(self.w), np.random.randint(self.h)
        self.i, self.j = i, j
        x0 = self.I0[i, j]
        x = self.I[i, j]
        r = 16
        dx = np.random.randint(-min(r, x), min(r, 255 - x) + 1)
        x1 = x + dx
        dE = (x0 - x1 if x1 < x0 else x1 - x0) - (x0 - x if x < x0 else x - x0)
        self.dx = dx
        self.dE = dE
        self.Eν = self.E + dE
    def accept(self):
        self.I[self.i, self.j] += self.dx
        self.E = self.Eν


Ls = range(1, 11, 2)
wlresults = [wanglandau(StatisticalImage(128 * np.ones((L, L), dtype=int)),
                        M=1_000_000, N=127*L*L + 1, E0=0, Ef=127*L*L)
             for L in Ls]


import pickle
with open('wlresults.pickle', 'wb') as f:
    pickle.dump(list(Ls), f)
    pickle.dump(wlresults, f)


L = Ls[3]
wlEs, S, H = wlresults[3]
L


# Look at the histogram to see how the last WL iteration went.

plt.plot(wlEs / L**2, H)
plt.xlabel("E / N")
plt.ylabel("Visits");


# Fit a spline to interpolate and optionally clean up noise, giving WL g's up to a normalization constant.

gspl = interpolate.splrep(wlEs, S, s=0*np.sqrt(2))
wlgsC = np.exp(interpolate.splev(wlEs, gspl) - min(S))


# ### Exact solution

# The exact density of states for uniform values. This covers the all gray and all black/white cases. Everything else (normal images) are somewhere between. The gray is a slight approximation: the ground level is not degenerate, but we say it has degeneracy 2 like all the other sites. For the numbers of sites and values we are using, this is insignificant.

def bw_g(E, N, M, exact=True):
    return sum((-1)**k * special.comb(N, k, exact=exact) * special.comb(E + N - 1 - k*(M + 1), E - k*(M + 1), exact=exact)
        for k in range(int(E / M) + 1))
def gray_g(E, N, M, exact=True):
    return 2 * bw_g(E, N, M, exact=exact)


# We only compute to halfway since $g$ is symmetric and the other half's large numbers cause numerical instability.

def reflect(a):
    return np.hstack([a[:-2], a[-1], a[-2::-1]])
def gray_gs(N, M):
    Es = np.arange(N*M + 1)
    gs = np.vectorize(gray_g)(np.arange(1 + N*M / 2), N, M, exact=False)
    return Es, reflect(gs)


Es, gs = gray_gs(N=L**2, M=2**7 - 1)


# Renormalize the WL result

wlgs = wlgsC * (gs[len(gs) // 2] / wlgsC[len(wlgsC) // 2])


# Compare the exact result to the WL result.

plt.plot(wlEs / len(wlEs), np.log(wlgs), label='WL')
plt.plot(Es / len(Es), np.log(gs), label='Exact')
plt.xlabel('E / N (symmetric)')
plt.ylabel('ln g')
plt.title('L = {}'.format(L))
plt.legend();


# Presumably all of the densities of states for different images fall in the region between the all-gray and all-black/white curves.

bwEs, bwgs = gray_gs(N=L**2, M=2**8 - 1)
bwgs /= 2 # Undo gray_gs degeneracy


plt.plot(bwEs / len(bwEs), np.log(bwgs), 'black', label='BW')
plt.plot(Es / len(Es), np.log(gs), 'gray', label='Gray')
plt.xlabel('E / N (symmetric)')
plt.ylabel('ln g')
plt.title('L = {}'.format(L))
plt.legend();


# ### Calculating canonical ensemble averages

class CanonicalEnsemble:
    def __init__(self, Es, gs, name):
        self.Es = Es
        self.gs = gs
        self.name = name
    def Z(self, β):
        return np.sum(self.gs * np.exp(-β * self.Es))
    def average(self, f, β):
        return np.sum(f(self) * self.gs * np.exp(-β * self.Es)) / self.Z(β)
    def energy(self, β):
        return self.average(lambda ens: ens.Es, β)
    def energy2(self, β):
        return self.average(lambda ens: ens.Es**2, β)
    def heat_capacity(self, β):
        return self.energy2(β) - self.energy(β)**2
    def free_energy(self, β):
        return -np.log(self.Z(β)) / β
    def entropy(self, β):
        return β * self.energy(β) + np.log(self.Z(β))


βs = [np.exp(k) for k in np.linspace(-8, 2, 500)]
wlens = CanonicalEnsemble(wlEs, wlgs, 'WL') # Wang-Landau results
xens = CanonicalEnsemble(Es, gs, 'Exact') # Exact
ensembles = [wlens, xens]


# Partition function

for ens in ensembles:
    plt.plot(-np.log(βs), np.log(np.vectorize(ens.Z)(βs)), label=ens.name)
plt.xlabel("ln kT")
plt.ylabel("Log partition function")
plt.title('L = {}'.format(L))
plt.legend();


# Helmholtz free energy

for ens in ensembles:
    plt.plot(-np.log(βs), np.vectorize(ens.free_energy)(βs), label=ens.name)
plt.xlabel("ln kT")
plt.ylabel("Helmholtz free energy")
plt.title('L = {}'.format(L))
plt.legend();


# Heat capacity

for ens in ensembles:
    plt.plot(-np.log(βs), np.vectorize(ens.heat_capacity)(βs), label=ens.name)
plt.xlabel("ln kT")
plt.ylabel("Heat capacity")
plt.title('L = {}'.format(L))
plt.legend();


# Entropy

for ens in ensembles:
    plt.plot(-np.log(βs), np.vectorize(ens.entropy)(βs), label=ens.name)
plt.xlabel("ln kT")
plt.ylabel("Canonical entropy")
plt.title('L = {}'.format(L))
plt.legend();

