# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:13:11 2023

@author: Gavin
"""

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
    
    f1_num = results['ppv'] * results['sensitivity']
    f1_den = results['ppv'] + results['sensitivity']
    
    results['f1'] = 2 * (f1_num / f1_den) if f1_den > 0 else 0
    
    return results