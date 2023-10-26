# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:28 2023

@author: Gavin
"""

import random
import torch

import numpy as np
import torchvision.transforms.functional as F

def random_hflip(*tensors, p=0.2):
    if np.random.rand() <= p:
        
        def hflip(tensor):
            if not torch.is_tensor(tensor):
                return [F.hflip(t) for t in tensor]
            
            return F.hflip(tensor)
                
        
        tensors = [hflip(t) for t in tensors]
        
    return tensors



def random_vflip(*tensors, p=0.2):
    if np.random.rand() <= p:
        
        def vflip(tensor):
            if not torch.is_tensor(tensor):
                return [F.vflip(t) for t in tensor]
            
            return F.vflip(tensor)
        
        tensors = [vflip(t) for t in tensors]
        
    return tensors



def random_rotate(*tensors, p=0.2):
    if np.random.rand() <= p:
        angle = random.uniform(0, 360)
        
        def rotate(tensor):
            if not torch.is_tensor(tensor):
                return [F.rotate(t, angle) for t in tensor]
            
            return F.rotate(tensor, angle)
        
        tensors = [rotate(t) for t in tensors]
        
    return tensors



def random_roll(*tensors, max_shift=50, p=0.2):
    vert_roll = np.random.randint(0, max_shift)
    horiz_roll = np.random.randint(0, max_shift)
    
    if np.random.rand() <= p:
        def roll(tensor):
            if not torch.is_tensor(tensor):
                rolled_tensor = []
                for t in tensor:
                    t = t.roll(horiz_roll, -1)
                    t = t.roll(vert_roll, -1)
                
                    rolled_tensor.append(t)
                    
                return rolled_tensor
            
            tensor = tensor.roll(horiz_roll, -1)
            tensor = tensor.roll(vert_roll, -1)
            
            return tensor
        
        tensors = [roll(t) for t in tensors]
        
    return tensors