# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:10:53 2024

@author: Gavin
"""

import torch

from torch import nn    

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        
        avgpool = nn.AdaptiveAvgPool3d(1)
        
        fc1 = nn.Conv3d(in_channels, squeeze_channels, 1)
        fc2 = nn.Conv3d(squeeze_channels, in_channels, 1)
        
        activation = nn.ReLU(inplace=True)
        scale_activation = nn.Sigmoid()

        self.features = nn.Sequential(
            avgpool,
            fc1,
            activation,
            fc2,
            scale_activation
        )

        

    def forward(self, x):
        scale = self.features(x)
        
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