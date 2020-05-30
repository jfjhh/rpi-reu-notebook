#!/usr/bin/env python
# coding: utf-8

# # Fractal dimension regression

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from scipy import interpolate
from src.intensity_entropy import *
from src.kernels import *
plt.rcParams['image.cmap'] = 'inferno'


img = ImageOps.grayscale(Image.open('test.jpg'))
scale = max(np.shape(img))
data = np.array(img)
img


# ## Box-counting dimension

def boxdim(data):
    εs = np.linspace(2, min(np.shape(data)))
    boxes = [np.log(np.sum(mapblocks(
        ε, ε, lambda x: 1 if np.any(x) else 0, data))) for ε in εs]
    logεs = np.log(εs)
    endεs = logεs[[0, -1]]
    dimfit = np.polyfit(np.log(εs), boxes, 1) # [slope, intercept]
    plt.plot(endεs, dimfit[0]*endεs + dimfit[1])
    plt.plot(logεs, boxes, '+')
    plt.xlabel('Scale (ln ε)')
    plt.ylabel('Box count (ln N)')
    return dimfit[0]


boxdim(data)


sky = data.copy()
sky[sky < 128+32] = 0
Image.fromarray(sky)


boxdim(sky)


nosky = data.copy()
nosky[nosky < 128+64] = 0
Image.fromarray(nosky)


boxdim(nosky)


dots = data.copy()
dots[nosky < 128+64+16] = 0
Image.fromarray(dots)


boxdim(dots)


# ## Information dimension

# Figured out problem: discretized samples must still be normalized. I.e. integrate over $\varepsilon$-segment to produce value. See [Wikipedia information dimension](https://en.wikipedia.org/wiki/Information_dimension#Connection_to_Differential_Entropy).

def infodim(dist):
    spl = interpolate.splrep(range(len(dist)), dist, s=0) # s=2e-5
    εs = np.arange(5, len(dist) - 1, 5)
    logεs = np.log2(εs)
    endεs = logεs[[0, -1]]
    entropies = [shannon_entropy(
        interpolate.splev(np.arange(0, len(dist), ε), spl)) for ε in εs]
    dimfit = np.polyfit(np.log2(εs), entropies, 1) # [slope, intercept]
    plt.plot(endεs, dimfit[0]*endεs + dimfit[1])
    plt.plot(logεs, entropies, '+')
    plt.xlabel('Scale (lg ε)')
    plt.ylabel('Shannon entropy (bits)')
    return dimfit


dist = (10 / 256) * np.exp(-np.linspace(-5, 5, 256)**2 / 2) / np.sqrt(2*np.pi)
plt.plot(dist);


infodim(dist)


dist = intensity_distribution(img)
plt.plot(dist);

