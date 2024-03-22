# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:54 2023

@author: Gavin
"""

from optim import TopoLoss
from projectio import load_train, CHASEDB1Dataset

device = 'cpu'

loader = load_train(
    'CHASEDB1', 
    CHASEDB1Dataset, 
    {'tile_size': 96, 'load_limit': None, 'device': device}, 
    {'batch_size': 4}
)

from models.unet import UNet2D

model = UNet2D(4, 3).to(device)
criterion = TopoLoss(device=device)

import matplotlib.pyplot as plt

for xs, ys in loader:
    # preds = model(xs)
    
    # loss = criterion(preds, ys)
    
    # print(loss.item())
    
    xs = xs.permute(0, 2, 1, 3, 4)
    ys = ys.permute(0, 2, 1, 3, 4)
    
    xs = xs.reshape(-1, *xs.shape[-3:]).permute(0, 3, 2, 1)
    ys = ys.reshape(-1, *ys.shape[-3:]).permute(0, 3, 2, 1)
    
    for x, y in zip(xs, ys):
        plt.imshow(x)
        plt.show()
        plt.imshow(y[..., :1], cmap='gray')
        plt.show()
    
    print(xs.shape, ys.shape)