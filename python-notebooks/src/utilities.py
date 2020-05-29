#!/usr/bin/env python
# coding: utf-8

# # Utilities

import numpy as np


def uniform(n):
    return np.array([1] * n) / n

def rescale(data):
    b = np.max(data)
    a = np.min(data)
    return (b - data) / (b - a)

