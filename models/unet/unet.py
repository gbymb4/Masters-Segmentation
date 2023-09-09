# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:12:18 2023

@author: Gavin
"""

import torch

from torch import nn

class UNet2D(nn.Module):

    def __init__(self, channels, img_channels=1):
        super().__init__()

        self.b1 = UNetBlock2D(img_channels, channels)

        self.mp2 = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )

        self.b3 = UNetBlock2D(channels, channels * 2)

        self.mp4 = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )

        self.b5 = UNetBlock2D(channels * 2, channels * 4)

        self.mp6 = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )

        self.b7 = UNetBlock2D(channels * 4, channels * 8)

        self.mp8 = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )

        self.b9 = UNetBlock2D(channels * 8, channels * 16)

        self.up10 = nn.ConvTranspose3d(
            channels * 16,
            channels * 8,
            kernel_size=(1, 2, 2),
            padding=(0, 0, 0),
            stride=(1, 2, 2)
        )

        self.b11 = UNetBlock2D(channels * 16, channels * 8)

        self.up12 = nn.ConvTranspose3d(
            channels * 8,
            channels * 4,
            kernel_size=(1, 2, 2),
            padding=(0, 0, 0),
            stride=(1, 2, 2)
        )

        self.b13 = UNetBlock2D(channels * 8, channels * 4)

        self.up14 = nn.ConvTranspose3d(
            channels * 4,
            channels * 2,
            kernel_size=(1, 2, 2),
            padding=(0, 0, 0),
            stride=(1, 2, 2)
        )

        self.b15 = UNetBlock2D(channels * 4, channels * 2)

        self.up16 = nn.ConvTranspose3d(
            channels * 2,
            channels,
            kernel_size=(1, 2, 2),
            padding=(0, 0, 0),
            stride=(1, 2, 2)
        )

        self.b17 = UNetBlock2D(channels * 2, channels)
        self.cn17 = nn.Conv3d(
            channels,
            2,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.a17 = nn.Sigmoid()



    def forward(self, x):
        if len(x.shape) == 4:
            dim = 0
        if len(x.shape) == 5:
            dim = 1

        out1 = self.b1(x)
        out2 = self.b3(self.mp2(out1))
        out3 = self.b5(self.mp4(out2))
        out4 = self.b7(self.mp6(out3))
        out5 = self.b9(self.mp8(out4))

        out6 = self.b11(torch.cat((out4, self.up10(out5)), dim=dim))
        out7 = self.b13(torch.cat((out3, self.up12(out6)), dim=dim))
        out8 = self.b15(torch.cat((out2, self.up14(out7)), dim=dim))
        out9 = self.b17(torch.cat((out1, self.up16(out8)), dim=dim))

        out10 = self.a17(self.cn17(out9))

        return out10



class UNetBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cn1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1)
        )
        self.a1 = nn.ReLU(inplace=True)

        self.cn2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1)
        )
        self.a2 = nn.ReLU(inplace=True)



    def forward(self, x):
        out = self.a1(self.cn1(x))
        out = self.a2(self.cn2(out))

        return out
    
    
    
class UNet3D(UNet2D):

    def __init__(self, channels, img_channels=1):
        super().__init__(channels, img_channels)

        self.b1 = UNetBlock3D(img_channels, channels)
        self.b3 = UNetBlock3D(channels, channels * 2)
        self.b5 = UNetBlock3D(channels * 2, channels * 4)
        self.b7 = UNetBlock3D(channels * 4, channels * 8)
        self.b9 = UNetBlock3D(channels * 8, channels * 16)
        self.b11 = UNetBlock3D(channels * 16, channels * 8)
        self.b13 = UNetBlock3D(channels * 8, channels * 4)
        self.b15 = UNetBlock3D(channels * 4, channels * 2)
        self.b17 = UNetBlock3D(channels * 2, channels)



class UNetBlock3D(UNetBlock2D):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

        self.cn1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )

        self.cn2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )

