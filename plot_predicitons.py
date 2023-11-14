# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:57:29 2023

@author: Gavin
"""

import torch, sys, os

from PIL import Image
from pconfig import parse_config, prepare_config, OUT_DIR
from projectio import load_test
from postprocessing import full_postprocess, split_segs_markers


def main():
    args = sys.argv[1:]
    save_freq, dataset, id, *models = args
    
    testloader = None
    for model_name in models:
        model_save_root = config_fname = os.path.join(
            OUT_DIR, 
            dataset.lower(), 
            model_name,
            str(id)
        )
        
        config_fname = os.path.join(model_save_root, 'config.yaml')
        config_dict = parse_config(config_fname)
        config_tup = prepare_config(config_dict)
        
        seed, dataset, model_type, device, id, checkpoint_freq, *rest = config_tup
        model_kwargs, optim_kwargs, dataloader_kwargs, dataset_kwargs = rest
    
        model = load_model(model_type, model_kwargs, dataset, id)
        model.to(device)
        
        if testloader is None:
            testloader = load_test(dataset, dataset_kwargs, dataloader_kwargs)

        test_visuals_root = os.path.join(model_save_root, 'test_visualisations')
        if not os.path.isdir(test_visuals_root):
            os.mkdir(test_visuals_root)

        xs_visuals_root = os.path.join(test_visuals_root, 'xs')
        ys_segs_visuals_root = os.path.join(test_visuals_root, 'ys_segs')
        post_pred_segs_visuals_root = os.path.join(test_visuals_root, 'post_pred_segs')

        if not os.path.isdir(xs_visuals_root):
            os.mkdir(xs_visuals_root)
            
        if not os.path.isdir(ys_segs_visuals_root):
            os.mkdir(ys_segs_visuals_root)
            
        if not os.path.isdir(post_pred_segs_visuals_root):
            os.mkdir(post_pred_segs_visuals_root)

        exec_model(model, testloader, device, save_freq, test_visuals_root)



def exec_model(model, testloader, device, save_freq, test_visuals_root):
    img_num = 0
    for batch in testloader:
        xs, ys = batch
        
        xs = xs.to(device)
        ys = ys.to(device)
    
        pred = model(xs)
        post_pred = full_postprocess(pred)
    
        ys_segs, ys_markers = split_segs_markers(ys)
        post_pred_segs, post_pred_markers = split_segs_markers(post_pred)

        xs = xs.reshape(-1, xs.shape[-2:])
        xs = xs.transpose(0, 2, 1)
        
        ys_segs = ys_segs.reshape(-1, ys_segs.shape[-2:])
        ys_segs = ys_segs.transpose(0, 2, 1)
    
        post_pred_segs = post_pred_segs.reshape(-1, post_pred_segs.shape[-2:])
        post_pred_segs = post_pred_segs.transpose(0, 2, 1)
        
        for x, y, p in zip(xs, ys_segs, post_pred_segs):
            if img_num % save_freq == 0:
                x = Image.fromarray(x.detach().cpu().numpy())
                x.save(os.path.join(test_visuals_root, f'x{img_num}.png'))
                
                y = Image.fromarray(p.detach().cpu().numpy())
                y.save(os.path.join(test_visuals_root, f'y{img_num}.png'))
                
                p = Image.fromarray(p.detach().cpu().numpy())
                p.save(os.path.join(test_visuals_root, f'p{img_num}.png'))
            
            img_num += 1
    
    

def load_model(model_type, model_kwargs, dataset, id):
    state_dicts_path = os.path.join(
        OUT_DIR, 
        dataset.lower(), 
        model_type.__name__,
        str(id),
        'model_checkpoints',
    )

    state_dicts_fps = sorted(os.listdir(state_dicts_path), key=lambda x: int(x[7:-3]))
    state_dict_fp = os.join(state_dicts_path, state_dicts_fps[-1])

    model = model_type(**model_kwargs)
    model.load_state_dict(torch.load(state_dict_fp))
    model.eval()
    
    return model
    


if __name__ == '__main__':
    main()