#!/usr/bin/env python
# coding: utf-8

# # Kernels
# Generalized to arbitrary functions on subregions of images.

import numpy as np


def box(x, y, r):
    return np.s_[max(0, x-r) : x+r+1, max(0, y-r) : y+r+1]
def mapbox(r, f, a):
    return np.reshape([f(a[box(*i, r)]) for i in np.ndindex(np.shape(a))], np.shape(a))
def mapboxes(rs, f, a):
    return (mapbox(r, f, a) for r in rs)
def mapallboxes(f, a):
    return mapboxes(range(max(np.shape(a))), f, a)

