# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:10:45 2023

@author: Gavin
"""

from torch import nn
from models.bifpn import SE_BiFPN
from ._core import MARMEL2D, RRDL2D, MARMEL3D, RRDL3D

class BiSE_MARM_UDet2D(nn.Module):
    
    def __init__(self, channels, img_channels=1, height=5, length=2):
        super().__init__()
        
        self.marmel1 = MARMEL2D(img_channels, channels, 1)
        self.marmel2 = MARMEL2D(channels, channels * 2, 2)
        self.marmel3 = MARMEL2D(channels * 2, channels * 4, 2)
        self.marmel4 = MARMEL2D(channels * 4, channels * 8, 2)
        self.marmel5 = MARMEL2D(channels * 8, channels * 16, 2)
        
        self.rrdl6 = RRDL2D(channels * 16, channels * 8, 2)
        self.rrdl7 = RRDL2D(channels * 8, channels * 4, 2)
        self.rrdl8 = RRDL2D(channels * 4, channels * 2, 2)
        self.rrdl9 = RRDL2D(channels * 2, channels, 2)
        
        self.se_bifpn = SE_BiFPN(channels, height, length)
        
        self.cn10 = nn.Conv3d(
            channels,
            2,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            stride=(1, 1, 1)
        )
        self.a10 = nn.Sigmoid()
        

    
    def forward(self, x):
        out1 = self.marmel1(x)
        out2 = self.marmel2(out1)
        out3 = self.marmel3(out2)
        out4 = self.marmel4(out3)
        out5 = self.marmel5(out4)
        
        out6 = self.rrdl6(out5, out4)
        out7 = self.rrdl7(out6, out3)
        out8 = self.rrdl8(out7, out2)
        out9 = self.rrdl9(out8, out1)
        
        se_bifpn_out = self.se_bifpn(out9, out8, out7, out6, out5)
        
        out10 = self.a10(self.cn10(se_bifpn_out))
        
        return out10
    
    

class BiSE_MARM_UDet3D(BiSE_MARM_UDet2D):
    
    def __init__(self, channels, img_channels=1, height=5, length=2):
        super().__init__(channels, img_channels=1, height=5, length=2)
        
        self.marmel1 = MARMEL3D(img_channels, channels, 1)
        self.marmel2 = MARMEL3D(channels, channels * 2, 2)
        self.marmel3 = MARMEL3D(channels * 2, channels * 4, 2)
        self.marmel4 = MARMEL3D(channels * 4, channels * 8, 2)
        self.marmel5 = MARMEL3D(channels * 8, channels * 16, 2)
        
        self.rrdl6 = RRDL3D(channels * 16, channels * 8, 2)
        self.rrdl7 = RRDL3D(channels * 8, channels * 4, 2)
        self.rrdl8 = RRDL3D(channels * 4, channels * 2, 2)
        self.rrdl9 = RRDL3D(channels * 2, channels, 2)
        
    