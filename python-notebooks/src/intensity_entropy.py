#!/usr/bin/env python
# coding: utf-8

# # Intensity-level entropy

# Given a discrete random variable $X$ on a probability space $(\Omega, \mathcal{F}, P)$ with image $\chi = \mathrm{im}(X)$, the *Shannon entropy* is
# $$
# H = \sum_{x \in \chi} -P(x) \ln P(x).
# $$
# The *intensity-level entropy* is the Shannon entropy of the empirical distribution of intensity values.

import numpy as np


def shannon_entropy(h):
    """The Shannon entropy in bits"""
    return -sum(p*np.log2(p) if p > 0 else 0 for p in h)

def intensity_distribution(data):
    """The intensity distribution of 8-bit `data`."""
    hist, _ = np.histogram(data, bins=range(256+1), density=True)
    return hist

def intensity_entropy(data):
    """The intensity-level entropy of 8-bit image data"""
    return shannon_entropy(intensity_distribution(data))

def intensity_expected(f, data):
    """The intensity-distribution expected value of `f`."""
    return sum(p*f(p) for p in intensity_distribution(data))

