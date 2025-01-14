# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:13:06 2023

@author: Gavin
"""

import torch

import numpy as np
import scipy.ndimage as ndi
import torchvision.transforms as T

from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights
from skimage import feature
from ._util import compute_coulomb_weightmaps

class SpatialWeightedBCELoss:

    def __init__(
        self, 
        positive_weight, 
        positive_weight_frac=1,
        epochs=500,
        weight_power=5,
        epsilon=1e-7
    ):
        self.weight_frac = positive_weight_frac
        self.weight = positive_weight
        self.epochs = epochs
        self.weight_power = weight_power
        self.epsilon = epsilon
        


    def __call__(self, pred, true, epoch, dists_arrays=None, qs=None):
        true = (true > 0).long()
        
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(dim=0)
            true = true.unsqueeze(dim=0)

        shape = true.shape
        temp_shape = (shape[0], shape[1] * shape[2], *shape[-2:])

        pred = torch.clip(pred, self.epsilon, 1 - self.epsilon)

        positive = (self.weight_frac * (self.weight - 1) + 1) * true * torch.log(pred)
        negative = (1 - true) * torch.log(1 - pred)

        total = positive + negative
        
<<<<<<< Updated upstream
        power = (1 + (epoch / self.epochs) * (self.weight_power - 1)) 
        
        reduced_size = [s // 2 for s in true.shape[-2:]]
        reduced_true = T.Resize(reduced_size, interpolation=0)(true.reshape(temp_shape))
        reduced_true = reduced_true.unsqueeze(dim=2)
        
        weight_map = compute_coulomb_weightmaps(
            reduced_true,
            p=power, 
            dists_arrays=dists_arrays,
            qs=qs
        )
        
        upscale_size = true.shape[-2:]
        weight_map = T.Resize(upscale_size, interpolation=0)(weight_map.squeeze(dim=2))
        weight_map = weight_map.unsqueeze(dim=2)
=======
        power = (1 + ((epoch / self.epochs) ** (1 / 2)) * (self.weight_power - 1)) 
>>>>>>> Stashed changes
        
        loss_temp = (total * weight_map).sum()
        loss = (-1 / pred.shape[0]) * loss_temp

        return loss
    
    

class SoftDiceLoss:
    
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
        
        

    def __call__(self, pred, true):
        true = (true > 0).long()
        
        pred = pred.reshape(-1)
        true = true.reshape(-1)
        
        pred = torch.clip(pred, self.epsilon, 1 - self.epsilon)

        intersection = (pred * true).sum()
        dice_coefficient = (2.0 * intersection) / (pred.sum() + true.sum())

        dice_loss = 1.0 - dice_coefficient

        return dice_loss
    
    
    
class HardDiceLoss:
    
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
        
        

    def __call__(self, pred, true):
        true = (true > 0).long()
        
        pred = pred.reshape(-1)
        true = true.bool().reshape(-1)
        
        pred = pred > 0.5

        intersection = (pred & true).sum()
        dice_coefficient = (2.0 * intersection) / (pred.sum() + true.sum())

        dice_loss = 1.0 - dice_coefficient

        return dice_loss
    
    

class PerceptualR50Loss:
    
    def __init__(self, device='cpu'):
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.eval()
        backbone.to(device)
        
        backbone = nn.Sequential(*list(backbone._modules.values())[:-2])
        
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone
        self.criterion = nn.MSELoss()



    def __fit_channels(self, tensor):
        first_channel = tensor[:, 0:1]
        second_channel = tensor[:, 1:]
        
        extra_channel = (first_channel + second_channel) / 2
        fitted_tensor = torch.cat((tensor, extra_channel), dim=1)
        
        return fitted_tensor



    def __call__(self, pred, true):
        true = (true > 0).long()
        
        pred = pred.reshape(pred.shape[0], -1, *pred.shape[-2:])
        true = true.reshape(true.shape[0], -1, *true.shape[-2:]).float()
        
        if len(pred.shape) == 4 and pred.shape[0] == 1:
            pred = pred.reshape(1, -1, *pred.shape[-2:])
            true = true.reshape(1, -1, *true.shape[-2:])
            
        elif len(pred.shape) == 5 and pred.shape[1] == 1:
            pred = pred.reshape(pred.shape[0], -1, *pred.shape[-2:])
            true = true.reshape(true.shape[0], -1, *true.shape[-2:])
            
        pred = self.__fit_channels(pred)
        true = self.__fit_channels(true)
        
        pred_fmap = self.backbone(pred)
        true_fmap = self.backbone(true)
        
        return self.criterion(pred_fmap, true_fmap)
    
    
    
class CompositeLoss:
    
    def __init__(self, 
        positive_weight, 
        wbce_positive_frac=1,
        wbce_weight=1, 
        dice_weight=100, 
        perc_weight=1,
        epochs=500,
        weight_power=5,
        epsilon=1e-7,
        device='cpu'
    ):
        self.wbce_weight = wbce_weight
        self.dice_weight = dice_weight
        self.perc_weight = perc_weight
        
        self.wbce = self.__default_loss
        self.dice = self.__default_loss
        self.perceptual = self.__default_loss
        
        if self.wbce_weight > 0:
            self.wbce = SpatialWeightedBCELoss(
                positive_weight, 
                positive_weight_frac=wbce_positive_frac,
                epochs=epochs,
                weight_power=epochs,
                epsilon=epsilon
            )
        if self.dice_weight > 0:
            self.dice = SoftDiceLoss(epsilon=epsilon)
        if self.perc_weight > 0:
            self.perceptual = PerceptualR50Loss(device=device)
        
    
    
    def __default_loss(self, pred, true):
        return 0
    
    
    
    def __call__(self, pred, true, epoch, dists_arrays=None, qs=None):
        true = (true > 0).long()
        
        wbce = self.wbce_weight * self.wbce(pred, true, epoch, dists_arrays, qs)
        dice = self.dice_weight * self.dice(pred, true)
        perceptual = self.perc_weight * self.perceptual(pred, true)
        
        return wbce + dice + perceptual
    
    
'''    
def get_weight_maps(tensors):
    device = tensors.device
    dtype = tensors.dtype
    
    arrays = tensors.detach().cpu().numpy()
    arrays_shape = arrays.shape
    
    arrays = arrays.reshape((-1, *arrays_shape[-2:]))
    
    border_width = 4
    
    # based on implementation from: https://github.com/CIVA-Lab/U-SE-ResNet-for-Cell-Tracking-Challenge/blob/main/SW/train_codes/data.py
    def weight_map(im):
        borders = feature.canny(im, low_threshold=.1, use_quantiles=True)
        
        borders[:border_width, :] = 0
        borders[-border_width:, :] = 0
        borders[:, :border_width] = 0
        borders[:, -border_width:] = 0
        
        dist_im = ndi.distance_transform_edt(1 - borders)
        wdist = ((dist_im.max() - dist_im)/dist_im.max())
        
        return wdist
    
    weight_map_vec = np.vectorize(weight_map, signature='(n,m)->(n,m)')
    
    weight_maps = weight_map_vec(arrays)
    weight_maps = weight_maps.reshape(arrays_shape)
    weight_maps = torch.tensor(weight_maps).to(device)
    
    return weight_maps'''