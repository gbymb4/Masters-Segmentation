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
    BiSE_MARM_UDet2D, BiSE_MARM_UDet3D, Alt_BiSE_MARM_UDet2D
)
from projectio import (
    CTCDataset,
    DRIVEDataset,
    STAREDataset,
    IOSTARDataset,
    HRFDataset
)

__ctc_datasets__ = [
    'BF-C2DL-HSC', 'BF-C2DL-MuSC',
    'DIC-C2DH-HeLa', 'Fluo-C2DL-Huh7',
    'Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1',
    'Fluo-N2DH-SIM+', 'Fluo-N2DL-HeLa',
    'PhC-C2DH-U373', 'PhC-C2DL-PSC'
]

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
    
    if dataset in __ctc_datasets__:
        dataset_type = CTCDataset
    elif dataset == 'DRIVE':
        dataset_type = DRIVEDataset
    elif dataset == 'STARE':
        dataset_type = STAREDataset
    elif dataset == 'IOSTAR':
        dataset_type = IOSTARDataset
    elif dataset == 'HRFDataset':
        dataset_type = HRFDataset
    else:
        raise ValueError(f'invalid dataset "{dataset}" in config file')
    
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
    else:
        raise ValueError(f'invalid model type "{model_name}" in config file')

    device = deepcopy(config['device'])
    id = deepcopy(config['id'])
    checkpoint_freq = deepcopy(config['checkpoint_freq'])
    
    model_kwargs = deepcopy(config['model_arguments'])
    optim_kwargs = deepcopy(config['optimizer_arguments'])
    dataloader_kwargs = deepcopy(config['dataloader_arguments'])
    dataset_kwargs = deepcopy(config['dataset_arguments'])

    args = (seed, dataset, dataset_type, model, device, id, checkpoint_freq)
    kwargs = (model_kwargs, optim_kwargs, dataloader_kwargs, dataset_kwargs)

    out = (*args, *kwargs)

    return out