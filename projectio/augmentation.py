# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:28 2023

@author: Gavin
"""

import random

import numpy as np
import torchvision.transforms.functional as F

def random_hflip(xs, ys, p=0.2):
    if np.random.rand() <= p:
        xs = F.hflip(xs)
        ys = F.hflip(ys)
        
    return xs, ys



def random_vflip(xs, ys, p=0.2):
    if np.random.rand() <= p:
        xs = F.vflip(xs)
        ys = F.vflip(ys)
        
    return xs, ys



def random_rotate(xs, ys, p=0.2):
    if np.random.rand() <= p:
        angle = random.uniform(0, 360)
        
        xs = F.rotate(xs, angle)
        ys = F.rotate(ys, angle)
        
    return xs, ys



def random_roll(xs, ys, max_shift=50, p=0.2):
    vert_roll = np.random.randint(0, max_shift)
    horiz_roll = np.random.randint(0, max_shift)
    
    if np.random.rand() <= p:
        xs = xs.roll(horiz_roll, -1)
        xs = xs.roll(vert_roll, -2)
        
        ys = ys.roll(horiz_roll, -1)
        ys = ys.roll(vert_roll, -2)
        
    return xs, ys