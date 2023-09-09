# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:46 2023

@author: Gavin
"""

import os, time, sys, warnings
import torch, random

import numpy as np

from optim import DefaultOptimizer
from postprocessing import full_postprocess, split_segs_markers
from projectio import (
    load_train,
    load_valid,
    plot_and_save_metric,
    plot_and_save_visual,
    save_history_dict_and_model,
    last_checkpoint
)
from pconfig import (
    parse_config, 
    prepare_config,
    OUT_DIR
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def dump_metrics_plots(model, dataset, id, history):
    metrics_keys = list(history[0].keys())
    num_keys = len(metrics_keys)
    
    history_transpose = {key: [] for key in metrics_keys}
    for epoch_dict in history:
        for key, value in epoch_dict.items():
            history_transpose[key].append(value)

    for train_metric, valid_metric in zip(metrics_keys[:num_keys // 2], metrics_keys[num_keys // 2:]):
        train_vals = history_transpose[train_metric]
        valid_vals = history_transpose[valid_metric]
        
        metric = '_'.join(train_metric.split('_')[1:])
        
        print(f'plotting {metric} figure...')
        
        plot_fname = os.path.join(
            OUT_DIR, 
            dataset.lower(), 
            type(model).__name__,
            str(id),
            f'{metric}.pdf'
        )
        
        plot_and_save_metric(train_vals, valid_vals, metric, plot_fname)
    


def dump_visualisations(
        model,
        dataset, 
        id,
        loader, 
        device,
        plot_num=None
    ):
    num_saved = 0
    for batch in loader:
        xs, ys = batch
        
        xs = xs.to(device)
        ys = ys.to(device)
        
        preds = model(xs)
        post_preds = full_postprocess(preds)
        post_pred_segs, post_pred_markers = split_segs_markers(post_preds)
        
        xs = xs.reshape(-1, *xs.shape[-2:])
        ys = ys.reshape(-1, *ys.shape[-2:])
        preds = preds.reshape(-1, *preds.shape[-2:])
        post_preds = post_preds.reshape(-1, *post_preds.shape[-2:])
                
        for x, y, pred, post_pred in zip(xs, ys, preds, post_pred_segs):
            if plot_num is not None and num_saved >= plot_num: return
            
            y = y > 0
            
            plot_root = os.path.join(
                OUT_DIR, 
                dataset.lower(), 
                type(model).__name__,
                str(id),
                'visualisations'
            )
            
            if not os.path.isdir(plot_root): os.mkdir(plot_root)
            
            plot_fname = os.path.join(plot_root, f'visual{num_saved}.pdf')
            
            plot_and_save_visual(x, y, pred, post_pred, plot_fname)
            num_saved += 1


        
def main():
    warnings.simplefilter('ignore')
    
    config_fname = sys.argv[1]
    config_dict = parse_config(config_fname)
    config_tup = prepare_config(config_dict)
    
    seed, dataset, model_type, device, id, checkpoint_freq, *rest = config_tup
    model_kwargs, optim_kwargs, dataloader_kwargs, dataset_kwargs = rest
    
    set_seed(seed)
    
    model = model_type(**model_kwargs).train().to(device)
        
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    if id is None:
        id = int(time.time())
    
    # checkpointing will only work for id's that aren't None
    last_epoch, last_history = last_checkpoint(
        dataset, 
        model, 
        id, 
    )
        
    def checkpoint(history, epoch):
        if epoch % checkpoint_freq == 0:
            save_history_dict_and_model(dataset, model, id, config_dict, history, epoch)
            dump_metrics_plots(model, dataset, id, history)
    
    trainloader = load_train(dataset, dataset_kwargs, dataloader_kwargs)
    validloader = load_valid(dataset, dataset_kwargs, dataloader_kwargs)
        
    optim = DefaultOptimizer(seed, model, trainloader, validloader, device=device)
    history = optim.execute(**optim_kwargs, checkpoint_callback=checkpoint)
        
    save_history_dict_and_model(dataset, model, id, config_dict, history, len(history))
        
    dump_metrics_plots(model, dataset, id, history)
    dump_visualisations(model, dataset, id, validloader, device)
    
        
    
if __name__ == '__main__':
    main()