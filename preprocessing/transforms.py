# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:24:51 2023

@author: Gavin
"""

import torch

import numpy as np

from scipy import ndimage
from skimage import morphology, measure
from skimage.transform import resize as r
from .utils import find_closest_pairs, compute_centroids

def resize(img, height, width):
    img = img.astype(np.float64)
    
    resized = r(
        img, 
        (height, width, *img.shape[2:]),
        anti_aliasing=False
    ).astype(np.uint16)
                
    return resized



def tile_split(img, chunk_size):
    dtype = img.dtype
    in_shape = img.shape
    
    H, W, C = in_shape
    I, J = np.ceil(H / chunk_size), np.ceil(W / chunk_size)
    I, J = int(I), int(J)
    
    tiles = np.zeros(((I * J), chunk_size, chunk_size, C))
    
    c = 0
    for i in range(I):
        for j in range(J):
            tile = img[
                chunk_size*i : chunk_size*(i+1), 
                chunk_size*j : chunk_size*(j+1), 
                :
            ]
            
            tiles[c] = tile
            
            c += 1
    
    tiles = tiles.astype(dtype)
    
    return tiles



def lcm_pad(img, lcm):
    H, W, C = img.shape
    
    if H % lcm == 0 and W % lcm == 0: return img
    
    nH = H + (lcm - (H % lcm))
    nW = W + (lcm - (W % lcm))
        
    padded_img = np.zeros((nH, nW, C))
    padded_img[:H, :W, :C] = img
    
    return padded_img



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



def get_markers(imgs, erosion=20):
    dtype = np.float32
    imgs_shape = imgs.shape
    
    imgs = imgs.reshape((-1, *imgs_shape[-2:]))
    
    # based on implementation from: https://github.com/CIVA-Lab/U-SE-ResNet-for-Cell-Tracking-Challenge/blob/main/SW/train_codes/data.py
    def markers(im, erosion=erosion):
        lab = measure.label(im)
        markers = np.zeros_like(lab)
        
        for i in range(1, lab.max() + 1):
            mask = lab == i
            
            eroded_mask = morphology.binary_erosion(
                mask,
                np.ones((erosion, erosion))
            )
            
            markers[eroded_mask] = 1
            
        return markers.astype(dtype)

    markers_vec = np.vectorize(markers, signature='(n,m)->(n,m)')

    imgs_markers = markers_vec(imgs).astype(dtype)
    imgs_markers = imgs_markers.reshape(imgs_shape)
    
    return imgs_markers
    


def get_dummy_markers(arr, dim=1):
    dummy = np.zeros_like(arr)
    new_arr = np.concatenate((arr, dummy), axis=dim)
    
    return new_arr
    


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