# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:04:59 2023

@author: Gavin
"""

import os, json, torch

from pconfig import OUT_DIR

def last_checkpoint(dataset, model, id):
    out_root = os.path.join(
        OUT_DIR, 
        dataset.lower(), 
        type(model).__name__,
        str(id)
    )
    
    if not os.path.isdir(out_root):
        return 0, None
    
    checkpoints_dir = os.path.join(f'{out_root}/model_checkpoints')
    
    if not os.path.isdir(checkpoints_dir):
        return 0, None
    
    model_checkpoints = os.listdir(checkpoints_dir)
    model_checkpoints = sorted(
        model_checkpoints, 
        key=lambda x: int(x[6:-3])
    )
    
    latest_epoch =  int(model_checkpoints[-1][6:-3])
    latest_checkpoint = os.path.join(checkpoints_dir, model_checkpoints[-1])
    
    model.load_state_dict(torch.load(latest_checkpoint))
    
    with open(os.path.join(out_root, 'history.json')) as f:
        latest_history = json.load(f)
    
    return latest_epoch, latest_history