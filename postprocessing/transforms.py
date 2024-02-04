# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:45:01 2023

@author: Gavin
"""

import torch

import numpy as np

from scipy import ndimage

def split_segs_markers(pred):
    segs = pred[:, 0:1]
    markers = pred[:, 1:]
    
    return segs, markers



def threshold(pred, true_threshold=0.5):
    return (pred > true_threshold).bool()



def clean_outliers(pred, area_threshold=50):
    device = pred.device
    
    def clean_img(img):
        img_temp = img.detach().cpu().numpy()
        
        new_img_temp = np.zeros(img_temp.shape)    
        for i, slide in enumerate(img_temp[0]):
            labeled_slide, num_features = ndimage.label(slide)
    
            region_sizes = ndimage.sum(slide, labeled_slide, range(num_features + 1))
        
            labels_to_remove = np.where(region_sizes < area_threshold)[0]
        
            cleaned_slide = np.copy(slide)
            for label in labels_to_remove:
                cleaned_slide[labeled_slide == label] = 0
                
            new_img_temp[0, i] = cleaned_slide
            
        img = torch.tensor(new_img_temp).bool().to(device)
        
        return img
    
    if len(pred.shape) == 4:
        return clean_img(pred)
    if len(pred.shape) == 5:
        imgs = [clean_img(img) for img in pred]    
        return torch.stack(imgs, dim=0)
    
    
    
def full_postprocess(pred, true_threshold=0.5, area_threshold=50):
    pred = threshold(pred, true_threshold)
    pred = clean_outliers(pred, area_threshold)
    
    return pred



def stitch_tiles(imgs, out_res):    
    if isinstance(imgs, list):
        imgs = torch.cat(imgs, dim=0)
    
    device = imgs.device
    dtype = imgs.dtype
    
    in_shape = imgs.shape
    tile_size = in_shape[-1]
    
    H, W = out_res
    I, J = np.ceil(H / in_shape[-1]), np.ceil(W / in_shape[-2])
    I, J = int(I), int(J)
    
    stitched_threshold = I * J
    num_stitched_imgs = imgs.shape[0] // stitched_threshold
    
    stitched_imgs = np.zeros((
        num_stitched_imgs,
        *in_shape[1:-2],
        J * in_shape[-2],
        I * in_shape[-1]
    ))
    
    c = 0
    for i in range(I):
        for j in range(J):
            tile = imgs[c].detach().cpu().numpy()
            
            placement_dim = c // stitched_threshold
            
            stitched_imgs[
                placement_dim : placement_dim+1, 
                :, :,
                tile_size*j : tile_size*(j+1),
                tile_size*i : tile_size*(i+1)
            ] = tile
    
            c += 1
    
    stitched_imgs = stitched_imgs[..., :W, :H]
    stitched_imgs = torch.from_numpy(stitched_imgs).to(dtype)
    stitched_imgs = stitched_imgs.to(device)
    
    return stitched_imgs