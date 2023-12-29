# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:24:51 2023

@author: Gavin
"""

import numpy as np

from scipy import ndimage
from skimage import morphology, measure
from skimage.transform import resize as r
from numpy.lib.stride_tricks import sliding_window_view
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
    img = img.transpose(2, 1, 0)
    
    def chunk_channel(img_channel):
        chunked_channel = sliding_window_view(img_channel, (chunk_size, chunk_size))
        chunked_channel = chunked_channel[::chunk_size, ::chunk_size]
        chunked_channel = chunked_channel.reshape(-1, chunk_size, chunk_size)
            
        return chunked_channel
    
    chunk_channel_vec = np.vectorize(chunk_channel, signature='(n,m)->(i,x,y)')
    
    chunked_img = chunk_channel_vec(img)
    chunked_img = chunked_img.transpose(1, 3, 2, 0)
    
    return chunked_img



def lcm_pad(img, lcm):
    H, W, C = img.shape
    
    if H % lcm == 0 and W % lcm == 0: return img
    
    nH, nW = H, W
    
    if H % lcm != 0:
        nH = H + (H % lcm)
    if W % lcm != 0:
        nW = W + (W % lcm)
        
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