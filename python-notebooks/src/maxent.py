#!/usr/bin/env python
# coding: utf-8

# # Maximum-entropy reconstruction

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc


# Given a measurement $D$ of a system $I_0$, we wish to reconstruct $I_0$ from $D$. We take the Bayesian approach and find the $I$ that maximizes $P(I|D) \propto P(D|I)P(I)$, where the likelihood is related to the error in $D$ due to $I$ and the prior is $P(I) = \exp(\lambda S(I))$, where $S$ is some notion of the entropy of an image. If $D = T(I_0)$, then $P(D|I) = \exp(-E(T(I), D))$, for some metric $E$ on $D$'s. From the form of the objective $\ln P(D|I) + \ln(I) = -E(T(I), D) + \lambda S(I)$, we see that the entropy acts as a regularizer.
# 
# The astronomers have both $I$ and $D$ as intensity lists $\mathbb{Z}_N \to \mathbb{R}_{\ge 0}$, where
# $$
# S(I)
# = \sum_i I_i \ln I_i
# $$
# and
# $$
# E(D', D)
# = \sum_i \exp\left(-\frac{{(D_i' - D_i)}^2}{2\sigma_i} \right)
# $$
# (and the $\sigma_i$ are all the same?). The transformation $T$ is convolution with the point spread function of the telescope.
# 
# To perform the optimization, we need not only the functions $S$ and $E$, but also a procedure to modify candidate images.

def maxent_objective(D, I, λ): # For minimization
    return D.E(I.transform()) - λ*I.S


# Greedy for now, but can be something like simulated annealing
def maxent(D, I, λ = 1, N = 1_000_000, ε = 1e-8):
    f0 = np.inf
    f = maxent_objective(D, I, λ)
    i = 0
    while ε < f0 - f and i < N:
        I.propose()
        fν = maxent_objective(D, I, λ)
        if fν < f: # Greedy
            f0, f = f, fν
            I.accept()
        i += 1
        if i % (N // 20) == 0:
            print("Maxent: {} / {}".format(i, N))
        
    print("Maxent: i: {} / {}, Δf: {}".format(i, N, f0 - f))
    return I, i, f


# ## Example: 1D Point from Gaussian

class Gaussian:
    def __init__(self, μ=0, σ=1):
        self.μ = μ
        self.σ = σ
    def E(self, Gν):
        return (self.μ - Gν.μ)**2 + (self.σ - Gν.σ)**2

class Point:
    def __init__(self, x=0):
        self.x = x
        self.dx = 0
        self.S = self.entropy()
    def propose(self):
        self.dx = np.random.rand() - 0.5
        self.S = self.entropy()
    def accept(self):
        self.x += self.dx
        self.dx = 0 # Idempotence
    def entropy(self):
        return -(self.x - 10)**2 # Opposite from true max
    def transform(self):
        return Gaussian(μ = self.x + self.dx)


D = Gaussian(0, 1)
I = Point(9)
I0, _, _ = maxent(D, I, λ = 1, N = 1_000_000, ε = 1e-4)
I0.x


# Results are sort of near 5. The optimization is terrible, but you get the idea.

# ## Example: Image from PSF convolution (measurement)

class DImage:
    def __init__(self, I):
        self.I = I
    def E(self, Dν):
        return np.sum((self.I - Dν.I)**2)

class IImage:
    def __init__(self, I):
        self.I = I
        self.w, self.h = np.shape(I)
        self.i, self.j = 0, 0
        self.I0 = 0
        self.Iν = 0
        n = int(np.sqrt(self.w * self.h))
        self.G = signal.windows.gaussian(n // 10, n // 50)
        self.S = self.entropy()
    def propose(self):
        self.i, self.j = np.random.randint(self.w), np.random.randint(self.h)
        I0 = self.I[self.i, self.j]
        self.I0 = I0
        self.Iν = np.random.randint(256)
        self.S = self.entropy()
    def accept(self):
        self.I[self.i, self.j] = self.Iν
    def entropy(self):
        return -np.sum(np.log(self.I + 1)) + self.I0*np.log(self.I0 + 1) - self.Iν*np.log(self.Iν + 1)
    def transform(self):
        Iν = self.I.copy()
        Iν[self.i, self.j] = self.Iν
        return DImage(signal.sepfir2d(Iν, self.G, self.G))


I = IImage(misc.ascent()[250:350,250:350])
plt.imshow(I.I, cmap='gray');


plt.imshow(I.transform().I, cmap='gray');


I = IImage(misc.ascent()[250:350,250:350])
Iguess = IImage(128 * np.ones((I.w, I.h), dtype=int))
I0, _, _ = maxent(I.transform(), Iguess, λ = 1e-9, N = 1_000_000, ε = 1e-20) # Just do the max iterations


plt.imshow(I0.I, cmap='gray');


plt.imshow(I0.I, cmap='gray');

