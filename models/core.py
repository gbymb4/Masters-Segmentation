# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:09:35 2023

@author: Gavin
"""

import torch 

from torch import nn

# based on implementation from https://github.com/TsukamotoShuchi/RCNN/blob/master/rcnnblock.py
class RCL2D(nn.Module):
    
    def __init__(self, channels, steps=4):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), 
            padding=(0, 1, 1), 
            bias=False
        )
        self.bn = nn.ModuleList([nn.BatchNorm3d(channels) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps

        self.shortcut = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), 
            padding=(0, 1, 1), bias=False
        )



    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.conv(x)
            else:
                z = self.conv(x) + self.shortcut(rx)
            x = self.relu(z)
            x = self.bn[i](x)
        return x
    
    

class RR2DBlock(nn.Module):
    
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        
        self.rcl1 = RCL2D(channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.a1 = nn.ReLU(inplace=True)
        
        self.rcl2 = RCL2D(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.a2 = nn.ReLU(inplace=True)
        
        
        
    def forward(self, x):
        out1 = self.a1(self.bn1(self.rcl1(x)))
        out2 = self.rcl2(out1)
        
        out3 = self.bn2(out2 + x)
        out4 = self.a2(out3)
        
        return out4



class RREL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__()
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = RR2DBlock(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        
    
    
    def forward(self, x):
        return self.features(x)
    
    
    
class RRDL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, dec_ratio):
        super().__init__()
        
        if dec_ratio == 3:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 3, 3),
                padding=(0, 0, 0)
            )
        elif dec_ratio == 2:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1)
            )
        else:
            raise ValueError('dec_ratio other than 2 or 3 is not supported')
        
        self.a_1 = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm3d(out_channels)
        self.cn = nn.Conv3d(
            out_channels * 2,
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        self.a_2 = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm3d(out_channels)
        self.b = RR2DBlock(out_channels)
        
        
        
    def forward(self, x, skip):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0
        
        out = torch.cat((self.bn_1(self.a_1(self.dcn(x))), skip), dim=dim)
        out = self.b(self.bn_2(self.a_2(self.cn(out))))

        return out



class RCL3D(RCL2D):
    
    def __init__(self, channels, steps=4):
        super().__init__(channels, steps=4)
        
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1), 
            bias=False
        )

        self.shortcut = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1), 
            bias=False
        )
    
    
    
class RR3DBlock(RR2DBlock):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.rcl1 = RCL3D(channels)
        self.rcl2 = RCL3D(channels)
    
    
    
class RREL3D(RREL2D):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__(in_channels, out_channels, enc_ratio)
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = RR3DBlock(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
    
    
    
class RRDL3D(RRDL2D):
    
    def __init__(self, in_channels, out_channels, dec_ratio):
        super().__init__(in_channels, out_channels, dec_ratio)
        
        if dec_ratio == 3:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 3, 3),
                padding=(0, 0, 0)
            )
        elif dec_ratio == 2:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1)
            )
        else:
            raise ValueError('dec_ratio other than 2 or 3 is not supported')
        
        self.cn = nn.Conv3d(
            out_channels * 2,
            out_channels, 
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        
        self.b = RR3DBlock(out_channels)
        
        