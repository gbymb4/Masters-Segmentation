# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:13:11 2023

@author: Gavin
"""

import numpy as np

from skimage.measure import label
from scipy.ndimage import binary_hit_or_miss
from .loss import HardDiceLoss

def accuracy(pred, true):
    pred = pred.reshape(-1)
    true = (true > 0).reshape(-1)
    
    correct = (pred == true).sum().item()
    total = len(pred)
    
    return correct / total



def sensitivity(pred, true):
    pred = pred.reshape(-1)
    true = (true > 0).reshape(-1)
    
    true_positives = (pred & true).sum().item()
    positives = true.sum().item()
    
    return true_positives / positives if positives > 0 else 0



def specificity(pred, true):
    pred = pred.reshape(-1)
    true = (true > 0).reshape(-1)
    
    true_negatives = (~pred & ~true).sum().item()
    negatives = (~true).sum().item()
    
    return true_negatives / negatives if negatives > 0 else 0



def ppv(pred, true):
    pred = pred.reshape(-1)
    true = (true > 0).reshape(-1)
    
    true_positives = (pred & true).sum().item()
    model_positives = pred.sum().item()
    
    return true_positives / model_positives if model_positives > 0 else 0
    
    

def npv(pred, true):
    pred = pred.reshape(-1)
    true = (true > 0).reshape(-1)
    
    true_negatives = (~pred & ~true).sum().item()
    model_negatives = (~pred).sum().item()
    
    return true_negatives / model_negatives if model_negatives > 0 else 0



def hard_dice(pred, true, epsilon=1e-7):
    criterion = HardDiceLoss(epsilon=epsilon)
    
    return criterion(pred, true).item()



def iou(pred, true):
    pred = pred.reshape(-1)
    true = (true > 0).reshape(-1)

    intersection = (pred & true).sum().item()
    union = (pred | true).sum().item()
    
    return intersection / union if union > 0 else 0



def betti_error(pred, true):
    pred = pred.detach().cpu().numpy()
    true = (true > 0).detach().cpu().numpy()

    pred = pred.transpose(0, 4, 3, 2, 1)[..., 0, 0]
    true = true.transpose(0, 4, 3, 2, 1)[..., 0, 0]
    
    def compute_betti_error(p, t):
        p_b0, t_b0 = compute_betti_0(p), compute_betti_0(t)
        p_b1, t_b1 = compute_betti_1(p), compute_betti_1(t)
    
        error = np.abs(p_b0 - t_b0) + np.abs(p_b1 - t_b1)
    
        return error
    
    def compute_betti_0(img):
        labeled_img, num_labels = label(img, connectivity=2, return_num=True)

        chi = num_labels - (len(np.unique(labeled_img)) - 1)
        b0 = num_labels - chi
        
        return b0
    
    def compute_betti_1(img):
        skeleton = binary_hit_or_miss(img)
        _, num_skeleton_labels = label(skeleton, connectivity=2, return_num=True)
        b1 = num_skeleton_labels
    
        return b1
    
    errors = [compute_betti_error(p, t) for p, t in zip(pred, true)]
    avg_error = sum(errors) / len(errors)
    
    return avg_error



def compute_all_metrics(pred, true, epsilon=1-7):
    results = {}
    
    results['accuracy'] = accuracy(pred, true)
    results['sensitivity'] = sensitivity(pred, true)
    results['specificity'] = specificity(pred, true)
    results['fnr'] = 1 - results['sensitivity']
    results['fpr'] = 1 - results['specificity']
    results['ppv'] = ppv(pred, true)
    results['npv'] = npv(pred, true)
    results['hard_dice'] = hard_dice(pred, true, epsilon=1-7)
    results['iou'] = iou(pred, true)
    results['betti_error'] = betti_error(pred, true)
    
    f1_num = results['ppv'] * results['sensitivity']
    f1_den = results['ppv'] + results['sensitivity']
    
    results['f1'] = 2 * (f1_num / f1_den) if f1_den > 0 else 0
    
    return results