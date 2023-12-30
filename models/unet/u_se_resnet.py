# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:19:05 2023

@author: Gavin
"""

import segmentation_models_pytorch as smp

from torch import nn

class U_SE_Resnet2D(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        if 'img_channels' in kwargs.keys():
            self.channels = kwargs['img_channels']
        else:
            self.channels = 1
        
        self.features = smp.Unet(encoder_name='se_resnet50', *args, **kwargs)
        
        
    
    def forward(self, x):
        shape = x.shape
        
        x = x.reshape(-1, self.channels, *shape[-2:])
        
        y = self.features(x)
        y = y.reshape(shape[0], 2, *shape[2:])
        
        return y