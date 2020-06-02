#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:37:30 2020

@author: augustinjose
"""

import numpy as np
from PIL import Image

def kernal(sigma, size):
    x = int((size+1)/2)
    d1 = 1/(sigma*((2*np.pi)**0.5))
    d2 = 2*(sigma**2)
    kernal = np.array([])
    for i in range(-x+1, x):
        g = d1*np.exp(-((i**2)/d2))
        kernal = np.append(kernal, g)
    return kernal

def openImage(image):
    image = Image.open(image)
    image = np.asarray(image)
    return image