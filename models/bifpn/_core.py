# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:43:54 2023

@author: Gavin
"""

import torch

from torch import nn

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Conv3d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv3d(squeeze_channels, in_channels, 1)
        
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

        

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        
        return scale * x
    
    
    
class SEFusion(nn.Module):
    
    def __init__(self, channels, num_tensors):
        super().__init__()
        
        self.channels = channels
        
        self.se = SqueezeExcitation(
            channels * num_tensors, 
            channels * num_tensors
        )
        
    
    
    def forward(self, *tensors):
        out = torch.cat(tensors, 1)
        out = self.se(out)
        out = torch.split(out, self.channels, dim=1)
        
        temp = out[0]
        for i in range(1, len(out)):
            temp = temp + out[i]
            
        out = temp
        
        return out
    
    
    
class ConvBlock(nn.Module):
    
    def __init__(self,
        in_channels, 
        out_channels, 
        kernel_size=(3, 3, 3),
        padding=(1, 1, 1),
        stride=(1, 1, 1),
    ):
        super().__init__()
        
        self.cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.a = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(out_channels)
        
        
    
    def forward(self, x):
        out = self.cn(x)
        out = self.a(out)
        out = self.bn(out)
        
        return out
    
    
    
class BiFPN_Head(nn.Module):
    
    def __init__(self, min_channels, size=5):
        super().__init__()
        
        self.size = size
        
        self.ups = nn.ModuleList([nn.ConvTranspose3d(
            min_channels * 2 ** (i + 1),
            min_channels, 
            kernel_size=(1, 2 ** (i + 1), 2 ** (i + 1)),
            padding=(0, 0, 0),
            stride=(1, 2 ** (i + 1), 2 ** (i + 1))
        ) for i in range(size - 1)])
        
        
        
    def forward(self, *fmaps):
        assert len(fmaps) == self.size
        
        up_fmaps = []
        for fmap, up in zip(fmaps[1:], self.ups):
            up_fmap = up(fmap)
            up_fmaps.append(up_fmap)
            
        out_fmap = fmaps[0]
        for fmap in up_fmaps:
            out_fmap += fmap
            
        return fmap