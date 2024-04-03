# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:13:06 2023

@author: Gavin
"""

import math
import torch

import gudhi as gd
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as mor

from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights
from skimage import feature

def compute_loss(criterion, preds, true, epoch):
    if isinstance(preds, list) or isinstance(preds, tuple):
        loss = 0
        
        for pred in preds:
            loss += criterion(pred, true, epoch)
            
        return loss
    
    return criterion(preds, true, epoch)



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
    
    

class TopoLoss:
    
    def __init__(self, topo_size=48, device='cpu'):
        self.device = device
        self.topo_size = topo_size
        
        
    
    def __call__(self, pred, true, _):
        loss = 0
            
        pred = pred.reshape(-1, *pred.shape[-2:])
        true = true.reshape(-1, *true.shape[-2:])
            
        for p, t in zip(pred, true):
            loss += self.__get_topo_loss(p, t, self.topo_size)        

        return loss
        
    
    
    # implementation from: https://github.com/HuXiaoling/TopoLoss/blob/master/topoloss_pytorch.py
    def __get_topo_loss(self, likelihood_tensor, gt_tensor, topo_size):
        """
        Calculate the topology loss of the predicted image and ground truth image 
        Warning: To make sure the topology loss is able to back-propagation, likelihood 
        tensor requires to clone before detach from GPUs. In the end, you can hook the
        likelihood tensor to GPUs device.
    
        Args:
            likelihood_tensor:   The likelihood pytorch tensor.
            gt_tensor        :   The groundtruth of pytorch tensor.
            topo_size        :   The size of the patch is used. Default: 100
    
        Returns:
            loss_topo        :   The topology loss value (tensor)
    
        """
    
        likelihood = likelihood_tensor.clone()
        gt = gt_tensor.clone()
    
        likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
        gt = torch.squeeze(gt).cpu().detach().numpy()
    
        topo_cp_weight_map = np.zeros(likelihood.shape)
        topo_cp_ref_map = np.zeros(likelihood.shape)
    
        for y in range(0, likelihood.shape[0], topo_size):
            for x in range(0, likelihood.shape[1], topo_size):
    
                lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                             x:min(x + topo_size, likelihood.shape[1])]
                gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                             x:min(x + topo_size, gt.shape[1])]
                
                if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
                if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue
    
                # Get the critical points of predictions and ground truth
                pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = self.__get_critical_points(lh_patch)
                pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = self.__get_critical_points(gt_patch)
    
                # If the pairs not exist, continue for the next loop
                if not(pairs_lh_pa): continue
                if not(pairs_lh_gt): continue
    
                res = self.__compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)
                
                if res == False: continue
                
                force_list, idx_holes_to_fix, idx_holes_to_remove = res
        
    
                if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                    for hole_indx in idx_holes_to_fix:
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                                bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                                bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                                dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                    for hole_indx in idx_holes_to_remove:
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                                bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                    likelihood.shape[1]):
                                topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                    lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                                dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                                0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                    likelihood.shape[1]):
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                    lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0
    
        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).to(self.device)
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).to(self.device)
    
        # Measuring the MSE loss between predicted critical points and reference critical points
        loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
        return loss_topo
    
    
    
    # implementation from: https://github.com/HuXiaoling/TopoLoss/blob/master/topoloss_pytorch.py
    def __get_critical_points(self, likelihood):
        """
        Compute the critical points of the image (Value range from 0 -> 1)
    
        Args:
            likelihood: Likelihood image from the output of the neural networks
    
        Returns:
            pd_lh:  persistence diagram.
            bcp_lh: Birth critical points.
            dcp_lh: Death critical points.
            Bool:   Skip the process if number of matching pairs is zero.
    
        """
        lh = 1 - likelihood
        lh_vector = np.asarray(lh).flatten()
        
        lh_cubic = gd.CubicalComplex(
            dimensions=[lh.shape[0], lh.shape[1]],
            top_dimensional_cells=lh_vector
        )
    
        Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
        pairs_lh = lh_cubic.cofaces_of_persistence_pairs()
        
        # If the paris is 0, return False to skip
        if (len(pairs_lh[0])==0): return 0, 0, 0, False
    
        # return persistence diagram, birth/death critical points
        pd_lh = np.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
        bcp_lh = np.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
        dcp_lh = np.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
        
        return pd_lh, bcp_lh, dcp_lh, True


    
    # implementation from: https://github.com/HuXiaoling/TopoLoss/blob/master/topoloss_pytorch.py
    def __compute_dgm_force(
            self, 
            lh_dgm, 
            gt_dgm, 
            pers_thresh=0.03, 
            pers_thresh_perfect=0.99, 
            do_return_perfect=False
        ):
        """
        Compute the persistent diagram of the image
    
        Args:
            lh_dgm: likelihood persistent diagram.
            gt_dgm: ground truth persistent diagram.
            pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
            between the local maximum critical point value with its neighouboring minimum critical point value.
            The value smaller than the persistent threshold should be filtered. Default: 0.03
            pers_thresh_perfect: The distance difference between two critical points that can be considered as
            correct match. Default: 0.99
            do_return_perfect: Return the persistent point or not from the matching. Default: False
    
        Returns:
            force_list: The matching between the likelihood and ground truth persistent diagram
            idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
            idx_holes_to_remove: The index of persistent points that require to remove for the following training
            process
    
        """
        if len(lh_dgm.shape) == 1: return False
        
        lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
        if (gt_dgm.shape[0] == 0):
            gt_pers = None;
            gt_n_holes = 0;
        else:
            gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
            gt_n_holes = gt_pers.size  # number of holes in gt
    
        if (gt_pers is None or gt_n_holes == 0):
            idx_holes_to_fix = list();
            idx_holes_to_remove = list(set(range(lh_pers.size)))
            idx_holes_perfect = list();
        else:
            # check to ensure that all gt dots have persistence 1
            tmp = gt_pers > pers_thresh_perfect
    
            # get "perfect holes" - holes which do not need to be fixed, i.e., find top
            # lh_n_holes_perfect indices
            # check to ensure that at least one dot has persistence 1; it is the hole
            # formed by the padded boundary
            # if no hole is ~1 (ie >.999) then just take all holes with max values
            tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
            lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
            if np.sum(tmp) >= 1:
                lh_n_holes_perfect = tmp.sum()
                idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
            else:
                idx_holes_perfect = list();
    
            # find top gt_n_holes indices
            idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];
    
            # the difference is holes to be fixed to perfect
            idx_holes_to_fix = list(
                set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))
    
            # remaining holes are all to be removed
            idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];
    
        # only select the ones whose persistence is large enough
        # set a threshold to remove meaningless persistence dots
        pers_thd = pers_thresh
        idx_valid = np.where(lh_pers > pers_thd)[0]
        idx_holes_to_remove = list(
            set(idx_holes_to_remove).intersection(set(idx_valid)))
    
        force_list = np.zeros(lh_dgm.shape)
        
        # push each hole-to-fix to (0,1)
        force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
        force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]
    
        # push each hole-to-remove to (0,1)
        force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                             math.sqrt(2.0)
        force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                             math.sqrt(2.0)

        if (do_return_perfect):
            return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect
        
        return force_list, idx_holes_to_fix, idx_holes_to_remove
    
    
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