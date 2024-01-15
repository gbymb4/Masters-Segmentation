# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:54 2023

@author: Gavin
"""

from models.unet import BiMAR_UDet2D

model = BiMAR_UDet2D(32, 1).cuda()

import torchsummary as ts

ts.summary(model, (1, 1, 256, 256))