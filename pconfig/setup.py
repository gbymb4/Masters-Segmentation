# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:11:51 2023

@author: Gavin
"""

import yaml

from copy import deepcopy
from models.unet import (
    UNet2D, UNet3D,
    R2UNet2D, R2UNet3D,
    R2UDet2D, R2UDet3D,
    SAR2UNet2D, SAR2UNet3D,
    U_SE_Resnet2D,
    MARM_UNet2D, MARM_UNet3D,
    BiSE_MARM_UDet2D, BiSE_MARM_UDet3D, Alt_BiSE_MARM_UDet2D,
    BiMAR_UDet2D, BiMAR_UDet3D,
    MAR_UNet2D, MAR_UNet3D,
    R2MidUDet2D, R2MidUDet3D
)
from models.sa2net import SA2Net2D



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
    elif model_name.lower() == 'r2unet2d':
        model = R2UNet2D
    elif model_name.lower() == 'r2unet3d':
        model = R2UNet3D
    elif model_name.lower() == 'r2udet2d':
        model = R2UDet2D
    elif model_name.lower() == 'r2udet3d':
        model = R2UDet3D
    elif model_name.lower() == 'sar2unet2d':
        model = SAR2UNet2D
    elif model_name.lower() == 'sar2unet3d':
        model = SAR2UNet3D
    elif model_name.lower() == 'u_se_resnet2d':
        model = U_SE_Resnet2D
    elif model_name.lower() == 'marm_unet2d':
        model = MARM_UNet2D
    elif model_name.lower() == 'marm_unet3d':
        model = MARM_UNet3D
    elif model_name.lower() == 'bise_marm_udet2d':
        model = BiSE_MARM_UDet2D
    elif model_name.lower() == 'bise_marm_udet3d':
        model = BiSE_MARM_UDet3D
    elif model_name.lower() == 'alt_bise_marm_udet2d':
        model = Alt_BiSE_MARM_UDet2D
    elif model_name.lower() == 'bimar_udet2d':
        model = BiMAR_UDet2D
    elif model_name.lower() == 'bimar_udet3d':
        model = BiMAR_UDet3D
    elif model_name.lower() == 'mar_unet2d':
        model = MAR_UNet2D
    elif model_name.lower() == 'mar_unet3d':
        model = MAR_UNet3D
    elif model_name.lower() == 'r2midudet2d':
        model = R2MidUDet2D
    elif model_name.lower() == 'r2midudet3d':
        model = R2MidUDet3D
    elif model_name.lower() == 'sa2net2d':
        model = SA2Net2D
    
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