# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:13:06 2023

@author: Gavin
"""

import torch

import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as mor

from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights
from skimage import feature

class SpatialWeightedBCELoss:

    def __init__(
        self, 
        positive_weight, 
        positive_weight_frac=1,
        epochs=500,
        weight_power=5,
        epsilon=1e-7,
        div_weight=1
    ):
        self.weight_frac = positive_weight_frac
        self.weight = positive_weight
        self.epochs = epochs
        self.weight_power = weight_power
        self.epsilon = epsilon
        self.div_weight = div_weight
        


    def __call__(self, pred, true, epoch):
        true = (true > 0).long()
        
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(dim=0)
            true = true.unsqueeze(dim=0)

        pred = torch.clip(pred, self.epsilon, 1 - self.epsilon)

        positive = (self.weight_frac * (self.weight - 1) + 1) * true * torch.log(pred)
        negative = (1 - true) * torch.log(1 - pred)

        total = (positive + negative) / (self.weight_frac * (self.weight - 1) + 1)
        
        power = (1 + ((epoch / self.epochs) ** (1 / 2)) * (self.weight_power - 1)) 
        
        weight_map = get_weight_maps(true, div_weight=self.div_weight)
        weight_map = weight_map ** power

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
        div_weight=1,
        epochs=500,
        weight_power=5,
        epsilon=1e-7,
        device='cpu'
    ):
        self.wbce_weight = wbce_weight
        self.dice_weight = 1 - self.wbce_weight
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
                epsilon=epsilon,
                div_weight=div_weight
            )
        if self.dice_weight > 0:
            self.dice = SoftDiceLoss(epsilon=epsilon)
        if self.perc_weight > 0:
            self.perceptual = PerceptualR50Loss(device=device)
        
    
    
    def __default_loss(self, pred, true):
        return 0
    
    
    
    def __call__(self, pred, true, epoch):
        true = (true > 0).long()
        
        wbce = self.wbce_weight * self.wbce(pred, true, epoch)
        dice = self.dice_weight * self.dice(pred, true)
        perceptual = self.perc_weight * self.perceptual(pred, true)
        
        return wbce + dice + perceptual
    
    
    
def get_weight_maps(tensors, div_weight=1):
    device = tensors.device   
    
    arrays = tensors.detach().cpu().numpy()
    arrays_shape = arrays.shape
    
    arrays = arrays.reshape((-1, *arrays_shape[-2:]))
    
    pad = 8
    
    W, H = arrays_shape[-2:]
    
    def weight_map(im):
        im_pad = np.zeros((W + pad, H + pad))
        im_pad[pad // 2 : W + pad // 2, pad // 2 : H + pad // 2] = im
        im_pad = mor.skeletonize(im_pad) | ndi.binary_erosion(im_pad, iterations=2)
        
        borders = feature.canny(im_pad, low_threshold=.1, use_quantiles=True)
        borders = borders[pad // 2 : W + pad // 2, pad // 2 : H + pad // 2]

        xy = np.indices((W, H))
        indices = np.argwhere(borders == 1)
        
        coulomb_vec_map = np.zeros((2, *borders.shape), dtype=float)

        dist_im = ndi.distance_transform_edt(1 - borders)
        wdist = ((dist_im.max() - dist_im)/dist_im.max())
        
        def coulomb_vec_channel(i, j):
            dists = np.zeros_like(borders, dtype=bool)
            dists[i, j] = 1
            dists = ndi.distance_transform_edt(1 - dists)
            
            forces = 1 / (dists ** 2)
            forces = forces.clip(0, 1)
            
            rel_xy = xy - np.array([i, j])[:, np.newaxis, np.newaxis]
            
            x_component = (rel_xy[0] / dists) * forces
            y_component = (rel_xy[1] / dists) * forces 
            
            return np.stack((x_component, y_component))
        
        for idx in indices:
            coulomb_vec_map += coulomb_vec_channel(*idx)
        
        def compute_divergence(vector_field):
            u, v = vector_field
        
            du_dx = np.gradient(u, axis=1)
            dv_dy = np.gradient(v, axis=0)
        
            div = du_dx + dv_dy
        
            return div
        
        coulomb_div_map = compute_divergence(coulomb_vec_map)
        
        coulomb_div_map[np.isnan(coulomb_div_map)] = 1
        coulomb_div_map[coulomb_div_map < 0] = 0
        coulomb_div_map[im > 0] = 0
        coulomb_div_map[borders] = 0
        coulomb_div_map = np.clip(coulomb_div_map, 0, 1)
        coulomb_div_map = np.sqrt(coulomb_div_map)
        
        weights = coulomb_div_map * div_weight
        weights = np.clip(weights + wdist, 0, 1)
        
        return weights
    
    weight_map_vec = np.vectorize(weight_map, signature='(n,m)->(n,m)')
    
    weight_maps = weight_map_vec(arrays)
    weight_maps = weight_maps.reshape(arrays_shape)
    weight_maps = torch.tensor(weight_maps).to(device)
    
    return weight_maps