# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:36:23 2023

@author: Gavin
"""

import torch

from torch import nn


'''
# Implementation based on: https://git.scc.kit.edu/kit-loe-ge/embedtrack/-/blob/master/embedtrack/models/erfnet.py?ref_type=heads
class Encoder(nn.Module):
    def __init__(self, num_classes, input_channels):
        super().__init__()
        
        self.initial_block = DownsamplerBlock(
            input_channels, 16
        )  # TODO input_channels = 1 (for gray-scale), 3 (for RGB)
        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))



    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output



# Implementation based on: https://git.scc.kit.edu/kit-loe-ge/embedtrack/-/blob/master/embedtrack/models/erfnet.py?ref_type=heads
class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)



    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)



# Implementation based on: https://git.scc.kit.edu/kit-loe-ge/embedtrack/-/blob/master/embedtrack/models/erfnet.py?ref_type=heads
class Decoder(nn.Module):
    def __init__(self, num_classes, n_init_features=128):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(n_init_features, 64))
        self.layers.append(non_bottleneck_1d(64, 0.0, 1))
        self.layers.append(non_bottleneck_1d(64, 0.0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0.0, 1))
        self.layers.append(non_bottleneck_1d(16, 0.0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )
        
        

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
'''