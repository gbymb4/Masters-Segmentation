# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:24:51 2023

@author: Gavin
"""

import numpy as np

from scipy import ndimage
from skimage import feature, morphology
from skimage.transform import resize as r
from skimage.measure import regionprops
from .utils import find_closest_pairs, compute_centroids

def resize(img, height, width):
    img = img.astype(np.float64)
    
    resized = r(
        img, 
        (height, width, *img.shape[2:]),
        anti_aliasing=False
    ).astype(np.uint16)
                
    return resized



# based on implementation from: https://github.com/CIVA-Lab/U-SE-ResNet-for-Cell-Tracking-Challenge/blob/main/SW/train_codes/data.py
def clip_limit(img, clim=0.01):

    if img.dtype == np.dtype(np.uint8):
        hist, *_ = np.histogram(
            img.reshape(-1),
            bins=np.linspace(0, 255, 255),
            density=True
        )
    elif img.dtype == np.dtype(np.uint16):
        hist, *_ = np.histogram(
            img.reshape(-1),
            bins=np.linspace(0, 65535, 65536),
            density=True
        )
        
    cumh = 0
    for i, h in enumerate(hist):
        cumh += h
        if cumh > 0.01:
            break
    
    cumh = 1
    for j, h in reversed(list(enumerate(hist))):
        cumh -= h
        if cumh < (1 - 0.01):
    
            break
    img = np.clip(img, i, j)
    
    return img



# based on implementation from: https://github.com/CIVA-Lab/U-SE-ResNet-for-Cell-Tracking-Challenge/blob/main/SW/train_codes/data.py
def normalize(arr):
    arr = clip_limit(arr)
    arr = arr.astype(np.float32)
    
    return (arr - arr.min()) / (arr.max() - arr.min())



# based on implementation from: https://github.com/CIVA-Lab/U-SE-ResNet-for-Cell-Tracking-Challenge/blob/main/SW/train_codes/data.py
def get_markers(lab, erosion):
    markers = np.zeros_like(lab)
    for i in range(1, lab.max() + 1):
        mask = lab == i
        
        eroded_mask = morphology.binary_erosion(
            mask,
            np.ones((erosion, erosion))
        )
        
        markers[eroded_mask] = 1
        
    return markers.astype(np.float32)



def resolve_seg_conflicts(gt_seg, st_seg, threshold=10):
    gt_lab, gt_num_labels = ndimage.label(gt_seg)
    st_lab, st_num_labels = ndimage.label(st_seg)
    
    gt_centroids = compute_centroids(gt_lab, gt_num_labels)
    st_centroids = compute_centroids(st_lab, st_num_labels)
    
    _, unmatched_centroids = find_closest_pairs(
        st_centroids, 
        gt_centroids, 
        threshold=threshold
    )
    
    if len(unmatched_centroids) > 0:
        for unmatched in unmatched_centroids:
            gt_seg[st_lab == unmatched] = 1