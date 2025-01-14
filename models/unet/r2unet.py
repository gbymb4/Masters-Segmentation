# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:56:01 2023

@author: Gavin
"""

from torch import nn
from ._core import RREL2D, RRDL2D, RREL3D, RRDL3D

class R2UNet2D(nn.Module):
    
    def __init__(self, channels, img_channels=1):
        super().__init__()
        
        self.rrel1 = RREL2D(img_channels, channels, 1)
        self.rrel2 = RREL2D(channels, channels * 2, 2)
        self.rrel3 = RREL2D(channels * 2, channels * 4, 2)
        self.rrel4 = RREL2D(channels * 4, channels * 8, 2)
        self.rrel5 = RREL2D(channels * 8, channels * 16, 2)
        
        self.rrdl6 = RRDL2D(channels * 16, channels * 8, 2)
        self.rrdl7 = RRDL2D(channels * 8, channels * 4, 2)
        self.rrdl8 = RRDL2D(channels * 4, channels * 2, 2)
        self.rrdl9 = RRDL2D(channels * 2, channels, 2)
        
        self.cn10 = nn.Conv3d(
            channels,
            2,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            stride=(1, 1, 1)
        )
        self.a10 = nn.Sigmoid()
        
    
    
    def forward(self, x):
        out1 = self.rrel1(x)
        out2 = self.rrel2(out1)
        out3 = self.rrel3(out2)
        out4 = self.rrel4(out3)
        out5 = self.rrel5(out4)
        
        out6 = self.rrdl6(out5, out4)
        out7 = self.rrdl7(out6, out3)
        out8 = self.rrdl8(out7, out2)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10
    


class R2UNet3D(R2UNet2D):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels=img_channels)
        
        self.rrel1 = RREL3D(img_channels, channels, 1)
        self.rrel2 = RREL3D(channels, channels * 2, 2)
        self.rrel3 = RREL3D(channels * 2, channels * 4, 2)
        self.rrel4 = RREL3D(channels * 4, channels * 8, 2)
        self.rrel5 = RREL3D(channels * 8, channels * 16, 2)
        
        self.rrdl6 = RRDL3D(channels * 16, channels * 8, 2)
        self.rrdl7 = RRDL3D(channels * 8, channels * 4, 2)
        self.rrdl8 = RRDL3D(channels * 4, channels * 2, 2)
        self.rrdl9 = RRDL3D(channels * 2, channels, 2)
        
