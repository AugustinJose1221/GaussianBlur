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
    k2 = np.array([])
    for i in range(-x+1, x):
        g = d1*np.exp(-((i**2)/d2))
        kernal = np.append(kernal, g)
        k2 = np.append(k2, [g,g,g])
    return kernal
    '''
    kernal = np.kron(kernal, k2)
    kernal = kernal.reshape(3,size,size)
    
    kernal = np.array([[[1,1,1], [4,4,4], [6,6,6], [4,4,4], [1,1,1]],
                      [[4,4,4], [16,16,16], [24,24,24], [16,16,16], [4,4,4]],
                      [[6,6,6], [24,24,24],[36,36,36] ,[24,24,24], [6,6,6]],
                      [[4,4,4], [16,16,16], [24,24,24], [16,16,16], [4,4,4]],
                      [[1,1,1], [4,4,4], [6,6,6], [4,4,4], [1,1,1]]])
  
    
    return kernal
    '''
print(kernal(1,3))
def openImage(image):
    image = Image.open(image)
    image = np.asarray(image)
    return image

def blur(image, level=5):
    gaussian = (1/256)*kernal(1, level)
    gl = len(gaussian)
    print(gl)
    frame = openImage(image)
    height = len(frame)
    width = len(frame[0])
    r = 0
    g = 0
    b = 0
    final_frame = np.zeros_like(frame, dtype=np.float32)
    for i in range(0,height):
        for j in range(0,width):
            if i+gl>height:
                break
            if j+gl>width:
                continue
            x = frame[i:i+gl, j:j+gl]
            y = x.flatten()
            g = gaussian.flatten()
            try:
                x = y*g
            except ValueError:
                print("VE")
                continue
            #'''
            for k in range(25):
                r = r+x[3*k]
                g = r+x[3*k+1]
                b = r+x[3*k+2]
            #'''      
            #final_frame[i][j] = x.sum() #np.array([r,g,b])
            final_frame[i][j] = [r,g,b]
    print(final_frame)
    img = Image.fromarray(final_frame, 'RGB')
    img.save('Output/Blur1.jpg')
    img.show()
    return True
'''
blur("img/test2.jpg")
print("Done")
'''