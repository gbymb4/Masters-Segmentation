# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:54 2023

@author: Gavin
"""

import torch, random, cv2

import segmentation_models_pytorch.utils.metrics

import matplotlib.pyplot as plt

import numpy as np

from projectio import *
from skimage import feature
import scipy.ndimage as ndi
from scipy.ndimage import convolve, gaussian_filter
import scipy.ndimage as ndi

seed = 0

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

d = STAREDataset('STARE', 'train', load_limit=1, tile_size=128)

count = 0
for img, seg in iter(d):
    img = img.detach().cpu().numpy()[:, 0, ...].transpose(2, 1, 0)
    seg = seg.detach().cpu().numpy()[:1, 0, ...].transpose(2, 1, 0)
    
    H, W, C = seg.shape
    
    pad = 8
    
    seg_pad = np.zeros((H + pad, W + pad, C))
    seg_pad[pad // 2 : H + pad // 2, pad // 2 : W + pad // 2, :] = seg    
    
    border = feature.canny(seg_pad[..., 0], low_threshold=.1, use_quantiles=True)
    border = border[pad // 2 : H + pad // 2, pad // 2 : W + pad // 2]
    
    indices = np.argwhere(border == 1)

    coulomb_vec_map = np.zeros((2, *border.shape), dtype=float)

    shape = border.shape
    #x, y = np.arange(shape[0]), np.arange(shape[1])
    
    xy = np.indices((shape[0], shape[1]))

    def coulomb_vec_channel(i, j):
        dists = np.zeros_like(border, dtype=bool)
        dists[i, j] = 1
        dists = ndi.distance_transform_edt(1 - dists)
        
        forces = 1 / (dists ** 2)
        forces = forces.clip(0, 1)
        
        rel_xy = xy - np.array([i, j])[:, np.newaxis, np.newaxis]
        
        x_component = (rel_xy[0] / dists) * forces
        y_component = (rel_xy[1] / dists) * forces 
        
        return np.stack((x_component, y_component))
    
    for idx in indices:
        coulomb_vec_map += coulomb_vec_channel(*idx)
    
    def compute_divergence(vector_field):
        # Extract the vector field components
        u = vector_field[0, :, :]
        v = vector_field[1, :, :]
    
        # Compute the partial derivatives along the x and y axes
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
    
        # Compute the divergence as the sum of partial derivatives
        divergence = du_dx + dv_dy
    
        return divergence
    
    coulomb_mag_map = np.sqrt((coulomb_vec_map[0] ** 2) + (coulomb_vec_map[1] ** 2))
    coulomb_div_map = compute_divergence(coulomb_vec_map)
    
    #coulomb_div_map = np.abs(coulomb_div_map)
    coulomb_div_map[np.isnan(coulomb_div_map)] = 1
    coulomb_div_map[coulomb_div_map < 0] = 0
    coulomb_div_map[seg[..., 0] > 0] = 0
    coulomb_div_map[border] = 0
    coulomb_div_map = np.clip(coulomb_div_map, 0, 1)
    #coulomb_div_map = (coulomb_div_map.max() - coulomb_div_map) / coulomb_div_map.max()
    
    print(coulomb_div_map.max(), coulomb_div_map.min())
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    axs[0].imshow(np.sqrt(coulomb_div_map))
    #axs[1].imshow(coulomb_mag_map)
    axs[1].imshow(seg, cmap='gray')
    
    plt.show()
    
    count += 1

    #print(img.shape, seg.shape)

    if count == 32: break

    