# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:02:19 2023

@author: Gavin
"""

from torch import nn
from ._core import BiFPN_Head
from .._core import SEFusion, ConvBlock

class SE_BiFPPNBlock(nn.Module):
    
    def __init__(self, min_channels, size=5):
        super().__init__()
        
        self.size = size
        
        self.downs = nn.ModuleList([ConvBlock(
            min_channels * 2 ** i,
            min_channels * 2 ** (i + 1),
            stride=(1, 2, 2)
        ) for i in range(size-1)])    
        
        self.ups = nn.ModuleList([nn.ConvTranspose3d(
            min_channels * 2 ** (i + 1),
            min_channels * 2 ** i,
            kernel_size=(1, 2, 2), 
            padding=(0, 0, 0),
            stride=(1, 2, 2)
        ) for i in range(size-1)])
        
        self.convs = nn.ModuleList([])
        for i in range(size):      
            convs = nn.ModuleList([
                ConvBlock(
                    min_channels * 2 ** i, 
                    min_channels * 2 ** i
                )
            ])
            
            if i == 0 or i == size - 1: 
                self.convs.append(convs)
                continue
        
            convs.extend(nn.ModuleList([
                ConvBlock(
                    min_channels * 2 ** i, 
                    min_channels * 2 ** i
                ),
                ConvBlock(
                    min_channels * 2 ** i, 
                    min_channels * 2 ** i
                )
            ]))
            
            self.convs.append(convs)
            
        self.fuses = nn.ModuleList([])
        for i in range(size):
            fuses = nn.ModuleList([SEFusion(min_channels * 2 ** i, 2)]) 
            
            if i == 0 or i == size - 1:
                self.fuses.append(fuses)
                continue
            
            fuses.append(SEFusion(min_channels * 2 ** i, 3))
            
            self.fuses.append(fuses)
            
        
        
    def forward(self, *fmaps):
        size = len(fmaps)
        
        assert size == self.size
        
        skip_convs = [self.convs[i][0](fmap) for i, fmap in enumerate(fmaps)]
        forward1_convs = [self.convs[i+1][1](fmap) for i, fmap in enumerate(fmaps[1:-1])]
        
        down_convs = []
        mid_fuses = []
        for i in range(size - 1):
            if i == 0:
                down_convs.append(self.downs[i](fmaps[i]))

            else:
                mid_fuses.append(
                    self.fuses[i][0](
                        forward1_convs[i-1], 
                        down_convs[-1]
                    )
                )
                
                down_convs.append(self.downs[i](mid_fuses[-1]))
            
        forward2_convs = [self.convs[i+1][2](fmap) for i, fmap in enumerate(mid_fuses)]
        
        end_fuses = []
        for i in range(size):
            if i == 0:
                end_fuses.append(
                    self.fuses[-1][0](
                        skip_convs[-1],
                        down_convs[-1]
                    )
                )
                
            elif i < size - 1:
                end_fuses.append(
                    self.fuses[::-1][i][1](
                        forward2_convs[::-1][i-1],
                        skip_convs[::-1][i],
                        self.ups[::-1][i-1](
                            end_fuses[-1]
                        )
                    )
                )
                
            else:
                end_fuses.append(
                    self.fuses[0][0](
                        skip_convs[::-1][i],
                        self.ups[::-1][i-1](
                            end_fuses[-1]
                        )
                    )  
                )
            
            
        out_fmaps = tuple(end_fuses[::-1])
            
        return out_fmaps
        
        
        
class SE_BiFPN(nn.Module):
    
    def __init__(self, min_channels, height=5, length=2):
        super().__init__()
        
        self.features = nn.ModuleList([SE_BiFPPNBlock(
            min_channels, 
            size=height
        ) for _ in range(length)])
        self.head = BiFPN_Head(min_channels, size=height)
    
        
    
    def forward(self, *fmaps):
        for bifpn_block in self.features:
            fmaps = bifpn_block(*fmaps)
            
        out = self.head(*fmaps)
        
        return out