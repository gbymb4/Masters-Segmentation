# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:12:57 2023

@author: Gavin
"""

import torch
import time, random

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from postprocessing import (
    full_postprocess, 
    threshold, 
    split_segs_markers,
    stitch_tiles
)
from projectio import get_dataset_res
from .loss import CompositeLoss
from .metrics import compute_all_metrics

class DefaultOptimizer:
    
    def __init__(self,
        seed: int,
        model: nn.Module, 
        train: DataLoader, 
        valid: DataLoader,
        device: str='cpu'
    ):
        super().__init__()
        
        self.seed = seed
        self.model = model
        self.train_loader = train
        self.valid_loader = valid
        self.device = device

        self.positive_weight = self.__compute_positive_weight()
    
    
    
    def __compute_positive_weight(self):
        positive_voxels = 0
        negative_voxels = 0

        for batch in self.train_loader:
            _, ys = batch
            
            for y in ys:
                y = y.cpu().detach().numpy()
    
                positive = y.sum()
                total = np.array(y.shape).prod()
                negative = total - positive
    
                positive_voxels += positive
                negative_voxels += negative

        return negative_voxels / positive_voxels
    
    
        
    def execute(self,
        epochs=100,
        start_epoch=0,
        lr=1e-5,
        valid_freq=10, 
        wbce_positive_frac=1,
        wbce_weight=1,
        dice_weight=100,
        perc_weight=1,
        div_weight=1,
        weight_power=5,
        verbose=True,
        checkpoint_callback=None,
        init_history = None
    ):
        if init_history is None:
            history = []
        else:
            history = init_history

        start = time.time()

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = CompositeLoss(
            self.positive_weight, 
            wbce_positive_frac=wbce_positive_frac,
            wbce_weight=wbce_weight,
            dice_weight=dice_weight,
            perc_weight=perc_weight,
            div_weight=div_weight,
            weight_power=weight_power,
            epochs=epochs,
            device=self.device
        )
        
        dataset_res = H, W = get_dataset_res(self.train_loader.dataset)
        tile_size = self.train_loader.dataset.tile_size
        I, J = np.ceil(H / tile_size), np.ceil(W / tile_size)
        count_threshold = int(I * J)
        
        print('#'*32)
        print('beginning BP training loop...')
        print('#'*32)
        
        for i in range(start_epoch, epochs):
            epoch = i + 1
            
            self.__reset_seed(self.seed + i)
            
            if verbose and i % 10 == 0:
                print(f'executing epoch {epoch}...', end='')
                
            history_record = {}
            
            train_num_imgs = 0
            train_loss = 0
            
            model = self.model.train()
            
            metrics_dict = {}

            for batch in self.train_loader:
                xs, ys = batch
                
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                batch_size = len(xs)

                optim.zero_grad()

                pred = model(xs)
                
                loss = criterion(pred, ys, epoch)
                loss.backward()
                train_loss = loss.item()

                post_pred = threshold(pred)
                
                ys_segs, ys_markers = split_segs_markers(ys)
                post_pred_segs, post_pred_markers = split_segs_markers(post_pred)
                
                # Betti Error metric not correctly implemented for training
                metric_scores = compute_all_metrics(post_pred_segs, ys_segs)
                        
                for name, score in metric_scores.items():
                    if name not in metrics_dict.keys():
                        metrics_dict[name] = score * batch_size
                    else:
                        metrics_dict[name] += score * batch_size

                train_num_imgs += len(xs)

                optim.step()
            
            history_record['train_loss'] = train_loss
            history_record['train_norm_loss'] = train_loss / train_num_imgs
            
            wavg_metrics = {
                f'train_{name}': w_score / train_num_imgs for name, w_score in metrics_dict.items()
            }
            
            history_record.update(wavg_metrics)

            if i % valid_freq == 0 or epoch == epochs:
                valid_num_slides = 0
                valid_loss = 0
    
                model = self.model.eval()
                
                metrics_dict = {}
    
                buffer_count = 0
                seg_buffer, pred_buffer = [], []
    
                for batch in self.valid_loader:
                    xs, ys = batch
                    
                    xs = xs.to(self.device)
                    ys = ys.to(self.device)
                    
                    batch_size = len(xs)

                    optim.zero_grad()

                    pred = model(xs)
                    
                    loss = criterion(pred, ys, epoch)
                    valid_loss = loss.item()

                    post_pred = full_postprocess(pred)
                    
                    ys_segs, ys_markers = split_segs_markers(ys)
                    post_pred_segs, post_pred_markers = split_segs_markers(post_pred)
                    
                    buffer_count += batch_size
                    full_buffers = buffer_count // count_threshold
                    
                    if full_buffers > 0:
                        buffer_tail = buffer_count - (count_threshold * full_buffers)
                        
                        batch_head = batch_size - buffer_tail

                        pred_buffer.append(post_pred_segs[:batch_head])
                        seg_buffer.append(ys_segs[:batch_head])
                        
                        f_post_pred_segs = stitch_tiles(pred_buffer, dataset_res)
                        f_ys_segs = stitch_tiles(seg_buffer, dataset_res)
                    
                        metric_scores = compute_all_metrics(f_post_pred_segs, f_ys_segs)
                                
                        for name, score in metric_scores.items():
                            if name not in metrics_dict.keys():
                                metrics_dict[name] = score  * full_buffers
                            else:
                                metrics_dict[name] += score * full_buffers

                        valid_num_slides += full_buffers
                        
                        pred_buffer = [post_pred_segs[batch_head:]]
                        seg_buffer = [ys_segs[batch_head:]]
                        
                        buffer_count = buffer_tail
                        
                    else:
                        pred_buffer.append(post_pred_segs)
                        seg_buffer.append(ys_segs)

                history_record['valid_loss'] = valid_loss
                history_record['valid_norm_loss'] = valid_loss / valid_num_slides
                
                wavg_metrics = {
                    f'valid_{name}': w_score / valid_num_slides for name, w_score in metrics_dict.items()
                }
                
                history_record.update(wavg_metrics)

            history.append(history_record)
            
            if checkpoint_callback is not None:
                checkpoint_callback(history, epoch)

            if verbose and i % 10 == 0 and epoch != epochs:
                print('done')
                print(f'epoch {epoch} training statistics:')
                print('\n'.join([f'->{key} = {value:.4f}' for key, value in history_record.items()]))
                print('-'*32)
            
        print('#'*32)
        print('finished BP training loop!')
        print('final training statistics:')
        print('\n'.join([f'->{key} = {value:.4f}' for key, value in history[-1].items()]))
        print('#'*32)

        end = time.time()

        print(f'total elapsed time: {end-start}s')
        
        return history
    
    
    
    def __reset_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)