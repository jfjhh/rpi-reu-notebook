#!/usr/bin/env python
# coding: utf-8

# # Color distributions

import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from src.utilities import *
from src.intensity_entropy import *


img = Image.open('test.jpg')
channels = img.split()
img


colors = ['r', 'g', 'b']
dists = [intensity_distribution(c) for c in channels]
for i, dist in enumerate(dists):
    plt.plot(dist, colors[i])
plt.show()


plt.plot(intensity_distribution(ImageOps.grayscale(img)), 'black');


channel_entropies = [intensity_entropy(c) for c in channels]
channel_entropies

