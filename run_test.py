# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:54 2023

@author: Gavin
"""

from projectio import CTCDataset

CTCDataset('PhC-C2DL-PSC', 'train', im_size=512, tile_size=96, load_limit=1)