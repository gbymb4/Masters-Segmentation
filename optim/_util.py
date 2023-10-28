# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:38:27 2023

@author: Gavin
"""

import torch

import numpy as np

from skimage import feature

def compute_coulomb_weightmaps(seg_maps, dists_arrays=None, qs=None, p=2):
    device = seg_maps.device
    dtype = seg_maps.dtype
    
    arrays = seg_maps.detach().cpu().numpy()
    arrays_shape = arrays.shape
    
    border_seg_maps = arrays.reshape((-1, *arrays_shape[-2:]))
    borders = compute_borders(border_seg_maps, arrays_shape)
    
    if dists_arrays is not None:
        
        def coulomb_func(instance_borders, dists_array_idx):
            dists_array = dists_arrays[dists_array_idx]
            q = qs[dists_array_idx]
            
            return compute_coulomb_array(instance_borders, dists_array, q, p=p)
    
        dists_array_idx = np.arange(len(dists_arrays)).astype(np.int32)
    
        coulomb_vec = np.vectorize(coulomb_func, signature='(n,m),()->(n,m)')
        coulomb = coulomb_vec(borders, dists_array_idx)
        
    else:
        
        def coulomb_func(instance_borders):
            return compute_coulomb_array(instance_borders, p=p)
    
        coulomb_vec = np.vectorize(coulomb_func, signature='(n,m)->(n,m)')
        coulomb = coulomb_vec(borders)
    
    if coulomb.max() > 0:    
        coulomb_norm = coulomb - coulomb.min() / (coulomb.max() - coulomb.min())
    else:
        coulomb_norm = coulomb

    coulomb_norm = coulomb_norm.reshape(arrays_shape)
    
    weight_maps = torch.tensor(coulomb_norm).type(dtype).to(device)

    return weight_maps



def compute_borders(seg_maps, init_shape, border_width=4):
    border_seg_maps = seg_maps.reshape((-1, *init_shape[-2:]))

    def border_func(seg_map_slice):
        edges = feature.canny(seg_map_slice, low_threshold=.1, use_quantiles=True)
        
        edges[:border_width, :] = 0
        edges[-border_width:, :] = 0
        edges[:, :border_width] = 0
        edges[:, -border_width:] = 0
        
        return edges
    
    borders_vec = np.vectorize(border_func, signature='(n,m)->(n,m)')
    borders = borders_vec(border_seg_maps).reshape(-1, *init_shape[-2:])
    borders = borders.astype(np.float32)
    
    return borders



def compute_coulomb_array(charges, dists_array=None, q=None, p=2):
    shape = charges.shape
    
    if dists_array is None:
        dists_array, q = compute_dists_array_from_borders(charges)
    
    elif dists_array.shape[-1] == 0:
        return np.zeros(dists_array.shape[:-1])
    
    p = np.full((*shape, dists_array.shape[-1]), p)
    p[dists_array < 0] = 1
    
    coulomb = np.divide(q, (dists_array ** p)).sum(axis=2)
    
    return coulomb.reshape(*shape)



def compute_dists_array_from_borders(charges):
    shape = charges.shape
    
    x, y = shape
    
    xy = np.meshgrid(np.arange(x), np.arange(y))
    xy = np.stack(xy, axis=-1)
    
    q_mask = charges == 1
    q_coords = xy[np.stack((q_mask, q_mask), axis=-1)].reshape(-1, 2)
    q = np.ones((*shape, len(q_coords))) 
    
    def compute_point_dists(origin):
        dist = compute_dists(origin, q_coords)
        
        return dist
    
    compute_point_vec = np.vectorize(compute_point_dists, signature='(2)->(n)')
    
    dists_array = compute_point_vec(xy)
    dists_array[dists_array == 0] = -1
    dists_array = dists_array.astype(np.float32)
    
    q[dists_array == -1] = 0
    
    return dists_array, q
    
    

def compute_dists(origin, coords):
    squared_diffs = np.sum((coords - origin[np.newaxis, :]) ** 2, axis=1)
    dist = np.sqrt(squared_diffs)
    
    return dist