# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:05:27 2023

@author: Gavin
"""

import torch

from torch import nn
from ._core import Encoder, Decoder



# based on implementatation from: https://git.scc.kit.edu/kit-loe-ge/embedtrack/-/blob/master/embedtrack/models/BranchedERFNet.py?ref_type=heads
class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()

        self.encoder = Encoder(sum(num_classes), input_channels)

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(Decoder(n))



    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            
            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2 : 2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2 : 2 + n_sigma].fill_(1)



    def forward(self, x):
        shape = x.shape
        
        x = x.reshape(-1, shape[1], *shape[-2:])
        
        out = self.encoder(x)
        out = torch.cat([decoder(out).reshape(*shape) for decoder in self.decoders], 1)

        return out

