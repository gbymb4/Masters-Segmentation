# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:24:03 2023

@author: Gavin
"""

from projectio.loading import (
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



def get_dataset_type(dataset):
    if dataset in __ctc_datasets__:
        dataset_type = CTCDataset
    elif dataset == 'DRIVE':
        dataset_type = DRIVEDataset
    elif dataset == 'STARE':
        dataset_type = STAREDataset
    elif dataset == 'IOSTAR':
        dataset_type = IOSTARDataset
    elif dataset == 'HRF':
        dataset_type = HRFDataset
    else:
        raise ValueError(f'invalid dataset "{dataset}" in config file')
        
    return dataset_type