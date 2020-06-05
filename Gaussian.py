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

def blur(image, level=5):
    gaussian = kernal(6, level)
    frame = openImage(image)
    height = len(frame)
    width = len(frame[0])
    conv = np.tile(0, 3)
    padding = int(level/2)
    blur_frame = np.tile(0, (height, width, 3))
    final_frame = np.tile(0, (height, width, 3))
    for i in range(height-len(gaussian)):
        for j in range(width-len(gaussian)):
            for k in range(len(gaussian)):
                conv = conv + frame[i][j+k]*gaussian[k]
            blur_frame[i][j] = conv
            conv = np.tile(0, 3)
    conv = np.tile(0, 3)
    for i in range(width-len(gaussian)):
        for j in range(height-len(gaussian)):
            for k in range(len(gaussian)):
                conv = conv + blur_frame[j+k][i]*gaussian[k]
            final_frame[j][i] = conv
            conv = np.tile(0, 3)
    
    img = Image.fromarray(final_frame, 'RGB')
    img.save('Output/Blur1.jpg')
    img.show()
    return True

blur("img/test2.jpg")
print("Done")
