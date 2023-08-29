# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:11:51 2023

@author: Gavin
"""

import yaml

from copy import deepcopy
from models.unet import UNet2D, UNet3D

def parse_config(fname: str) -> dict:
    with open(fname, 'r') as file:
        config = yaml.safe_load(file)

    return config



def prepare_config(
    config: dict
) -> tuple:
    seed = config['seed']
    dataset = deepcopy(config['dataset'])
    model_name = deepcopy(config['model'])
    
    if model_name.lower() == 'unet2d':
        model = UNet2D
    elif model_name.lower() == 'unet3d':
        model = UNet3D
    else:
        raise ValueError(f'invalid model type "{model_name}" in config file')

    device = deepcopy(config['device'])
    id = deepcopy(config['id'])
    checkpoint_freq = deepcopy(config['checkpoint_freq'])
    
    model_kwargs = deepcopy(config['model_arguments'])
    optim_kwargs = deepcopy(config['optimizer_arguments'])
    dataloader_kwargs = deepcopy(config['dataloader_arguments'])
    dataset_kwargs = deepcopy(config['dataset_arguments'])

    args = (seed, dataset, model, device, id, checkpoint_freq)
    kwargs = (model_kwargs, optim_kwargs, dataloader_kwargs, dataset_kwargs)

    out = (*args, *kwargs)

    return out