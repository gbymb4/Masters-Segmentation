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