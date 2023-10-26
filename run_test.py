# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:54 2023

@author: Gavin
"""

import segmentation_models_pytorch.utils.metrics

from optim import SpatialWeightedBCELoss

import torch

import numpy as np

from projectio import load_train

seed = 0

np.random.seed(seed)
torch.manual_seed(seed)

import random

random.seed(seed)

size = 128

pred = torch.zeros((4, 1, 1, size, size))
epoch = 1

l = SpatialWeightedBCELoss(1, weight_power=2)

dataset = 'PhC-C2DL-PSC'
dataset_kwargs = {
  'device': 'cpu',
  'im_size': size,
  'load_limit': 4
}

dataloader_kwargs = { 'batch_size': 4 }

train = load_train(dataset, dataset_kwargs, dataloader_kwargs)

x, y, dists, qs = next(iter(train))

true = y

print(l(pred, true, epoch, dists, qs))