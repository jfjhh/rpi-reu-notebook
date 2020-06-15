#!/usr/bin/env python
# coding: utf-8

# # Greedy Cubism
# Draw an image by greedily drawing cubes.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from scipy import signal, misc


img = Image.open("barbara.png")


def greedy_cubes(img, N):
    xs, ys = img.size
    art = Image.new(img.mode, (xs, ys))
    draw = ImageDraw.Draw(art)
    rmax = int(np.sqrt(xs*ys) / 10)
    r = rmax
    ε = 10
    for i in range(N):
        x = np.random.randint(xs)
        y = np.random.randint(ys)
        [np.mean(c) for c in cimg.split()]
        r = int(rmax * (1 - (i/(N+1))**2)) + 1
        box = [x - r, y - r, x + r, y + r]
        color = tuple(int(np.round(np.mean(c))) for c in img.crop(box=box).split())
        draw.ellipse(box, fill=color)
    return art


art = greedy_cubes(img, 10000)


art


cimg = Image.open('test.jpg')
cimg


art = greedy_cubes(cimg, 1000000)
art

