# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:46 2023

@author: Gavin
"""

import os, time, sys, warnings, json
import torch, random

import numpy as np

from optim import DefaultOptimizer, compute_all_metrics
from postprocessing import full_postprocess, split_segs_markers, stitch_tiles
from projectio import (
    load_train,
    load_valid,
    load_test,
    plot_and_save_metric,
    plot_and_save_visual,
    save_history_dict_and_model,
    last_checkpoint,
    get_dataset_type,
    get_dataset_res
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



def dump_test_metrics(model, testloader, dataset, id, device):
    dataset_res = H, W = get_dataset_res(testloader.dataset)
    tile_size = testloader.dataset.tile_size
    I, J = np.ceil(H / tile_size), np.ceil(W / tile_size)
    count_threshold = I * J
    
    test_num_slides = 0
    buffer_count = 0

    model = model.eval()
    
    history_record = {}
    seg_buffer, pred_buffer = [], []

    for batch in testloader:
        xs, ys = batch
        
        xs = xs.to(device)
        ys = ys.to(device)
        
        batch_size = len(xs)

        pred = model(xs)

        post_pred = full_postprocess(pred)
        
        ys_segs, ys_markers = split_segs_markers(ys)
        post_pred_segs, post_pred_markers = split_segs_markers(post_pred)
        
        buffer_count += batch_size
        full_buffers = buffer_count // count_threshold
        
        if full_buffers > 0:
            buffer_tail = buffer_count - (count_threshold * full_buffers)
            
            batch_head = batch_size - buffer_tail

            pred_buffer.append(post_pred_segs[:batch_head])
            seg_buffer.append(ys_segs[:batch_head])
            
            f_post_pred_segs = stitch_tiles(pred_buffer, dataset_res)
            f_ys_segs = stitch_tiles(seg_buffer, dataset_res)
        
            metric_scores = compute_all_metrics(f_post_pred_segs, f_ys_segs)
            
            for name, score in metric_scores.items():
                if name not in history_record.keys():
                    history_record[name] = score * full_buffers
                else:
                    history_record[name] += score * full_buffers

            test_num_slides += full_buffers
            
            pred_buffer = [post_pred_segs[batch_head:]]
            seg_buffer = [ys_segs[batch_head:]]
            
            buffer_count = buffer_tail
        
        else:
            pred_buffer.append(post_pred_segs)
            seg_buffer.append(ys_segs)
    
    history_record = {
        f'test_{name}': w_score / test_num_slides for name, w_score in history_record.items()
    }
    
    out_fname = os.path.join(
        OUT_DIR, 
        dataset.lower(), 
        type(model).__name__,
        str(id),
        'test.json'
    )

    with open(out_fname, 'w') as file:
        json.dump(history_record, file)



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
        
        ys_segs, ys_markers = split_segs_markers(ys)
        pred_segs, pred_markers = split_segs_markers(preds)
        post_pred_segs, post_pred_markers = split_segs_markers(post_preds)
        
        xs = xs.reshape(-1, *xs.shape[-2:])
        ys = ys.reshape(-1, *ys.shape[-2:])
        preds = preds.reshape(-1, *preds.shape[-2:])
        post_preds = post_preds.reshape(-1, *post_preds.shape[-2:])
                
        for x, y, pred, post_pred in zip(xs, ys_segs, pred_segs, post_pred_segs):
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
    
    dataset_type = get_dataset_type(dataset)
    
    set_seed(seed)
    
    model = model_type(**model_kwargs).train().to(device)
        
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except:
            print('model compilation failed due to error.')
    
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
    
    trainloader = load_train(dataset, dataset_type, dataset_kwargs, dataloader_kwargs)
    validloader = load_valid(dataset, dataset_type, dataset_kwargs, dataloader_kwargs)
    testloader = load_test(dataset, dataset_type, dataset_kwargs, dataloader_kwargs)
    
    optim = DefaultOptimizer(seed, model, trainloader, validloader, device=device)
    history = optim.execute(**optim_kwargs, checkpoint_callback=checkpoint)
        
    save_history_dict_and_model(dataset, model, id, config_dict, history, len(history))
        
    dump_test_metrics(model, testloader, dataset, id, device)
    dump_metrics_plots(model, dataset, id, history)
    dump_visualisations(model, dataset, id, validloader, device)
    
        
    
if __name__ == '__main__':
    main()