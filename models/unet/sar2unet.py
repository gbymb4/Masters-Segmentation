# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 14:32:06 2023

@author: Gavin
"""

from .r2unet import R2UNet2D, R2UNet3D
from .._core import AttentionBlock2D, AttentionBlock3D

class SAR2UNet2D(R2UNet2D):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.att2 = AttentionBlock2D(channels * 2)
        self.att3 = AttentionBlock2D(channels * 4)
        self.att4 = AttentionBlock2D(channels * 8)
        self.att5 = AttentionBlock2D(channels * 16)
        
    
    
    def forward(self, x):
        out1 = self.rrel1(x)
        
        out2_1 = self.rrel2(out1)
        out2_2 = self.att2(out2_1)
        out2_3 = out2_2 * out2_1
        out2_4 = out2_3 + out2_1
        
        out3_1 = self.rrel3(out2_4)
        out3_2 = self.att3(out3_1)
        out3_3 = out3_2 * out3_1
        out3_4 = out3_3 + out3_1
        
        out4_1 = self.rrel4(out3_4)
        out4_2 = self.att4(out4_1)
        out4_3 = out4_2 * out4_1
        out4_4 = out4_3 + out4_1
        
        out5_1 = self.drrel5(out4_4)
        out5_2 = self.att5(out5_1)
        out5_3 = out5_2 * out5_1
        out5_4 = out5_3 + out5_1
        
        out6 = self.rrdl6(out5_4, out4_4)
        out7 = self.rrdl7(out6, out3_4)
        out8 = self.rrdl8(out7, out2_4)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10
    
    
    
class SAR2UNet3D(R2UNet3D):
    
    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)
        
        self.att2 = AttentionBlock3D(channels * 2)
        self.att3 = AttentionBlock3D(channels * 4)
        self.att4 = AttentionBlock3D(channels * 8)
        self.att5 = AttentionBlock3D(channels * 16)
        
    
    
    def forward(self, x):
        out1 = self.rrel1(x)
        
        out2_1 = self.rrel2(out1)
        out2_2 = self.att2(out2_1)
        out2_3 = out2_2 * out2_1
        out2_4 = out2_3 + out2_1
        
        out3_1 = self.rrel3(out2_4)
        out3_2 = self.att3(out3_1)
        out3_3 = out3_2 * out3_1
        out3_4 = out3_3 + out3_1
        
        out4_1 = self.rrel4(out3_4)
        out4_2 = self.att4(out4_1)
        out4_3 = out4_2 * out4_1
        out4_4 = out4_3 + out4_1
        
        out5_1 = self.drrel5(out4_4)
        out5_2 = self.att5(out5_1)
        out5_3 = out5_2 * out5_1
        out5_4 = out5_3 + out5_1
        
        out6 = self.rrdl6(out5_4, out4_4)
        out7 = self.rrdl7(out6, out3_4)
        out8 = self.rrdl8(out7, out2_4)
        out9 = self.rrdl9(out8, out1)
        
        out10 = self.a10(self.cn10(out9))
        
        return out10