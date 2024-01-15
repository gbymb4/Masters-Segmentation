# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:43:54 2023

@author: Gavin
"""

from torch import nn    
      
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