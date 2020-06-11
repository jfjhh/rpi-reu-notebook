#!/usr/bin/env python
# coding: utf-8

# # The Wang-Landau algorithm (density of states)
# We determine thermodynamic quantities from the partition function by obtaining the density of states from a simulation.

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# The test system is the 2d Ising model.

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

# A Wang-Landau algorithm, with quantities as logarithms and with monte-carlo steps proportional to $f^{-1/2}$ (a "Zhou-Bhat schedule").

def flat(H, tol = 0.1):
    """Determines if an evenly-spaced histogram is approximately flat."""
    Hμ = np.mean(H)
    Hf = np.max(H)
    H0 = np.min(H)
    return Hf / (1 + tol) < Hμ < H0 / (1 - tol)


# Note: some parameters are hardcoded for testing
def density_sim(system):
    randint = np.random.randint
    rand = np.random.rand
    exp = np.exp
    
    # Parameters
    M = 1_000_000 # Monte carlo step scale
    ε = 1e-6
    logftol = np.log(1 + ε)
    logf0 = 1
    N = int(32**2 / 20) # Energy bins
    E0 = -32**2 / 4
    Ef = 32**2 / 4

    ΔE = (Ef - E0) / (N - 1)
    fiters = int(np.ceil(np.log2(logf0) - np.log2(logftol)))
    fiter = 0
    mciters = 0
    Es = np.linspace(E0, Ef, N)
    S = np.zeros(N) # Set all initial g's to 1
    H = np.zeros(N, dtype=int)
    logf = logf0
    # Linearly bin the energy
    i = max(0, min(N - 1, int(round((N - 1) * (system.E - E0) / (Ef - E0)))))
    print("ΔE = {}".format(ΔE))
    while logftol < logf:
        H[:] = 0
        logf /= 2
        iters = 0
        niters = int((M + 1) * exp(-logf / 2))
        fiter += 1
        while not flat(H[:-1]) and iters < niters:
            system.propose()
            Eν = system.Eν
            j = max(0, min(N - 1, int(round((N - 1) * (Eν - E0) / (Ef - E0)))))
            if E0 - ΔE/2 <= Eν <= Ef + ΔE/2 and (S[j] < S[i] or rand() < exp(S[i] - S[j])):
                system.accept()
                i = j
            H[i] += 1
            S[i] += logf
            iters += 1
        mciters += iters
        print("f: {} / {}\t({} / {})".format(fiter, fiters, iters, niters))
    
    print("Done: {} total MC iterations.".format(mciters))
    return Es, S, H


isingn = 32
sys = Ising(isingn)
Es, S, H = density_sim(sys);


# ## Calculating canonical ensemble averages

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

βs = [np.exp(k) for k in np.linspace(-5, 0, 200)]
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

