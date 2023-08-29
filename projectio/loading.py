# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:08 2023

@author: Gavin
"""

import os
import torch

import numpy as np
import skimage.io as sio
import torchvision.transforms.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pconfig import DATA_ROOT
from preprocessing import resolve_seg_conflicts, normalize, resize
from .augmentation import (
    random_hflip,
    random_vflip,
    random_rotate,
    random_roll
)

def load_train(dataset, dataset_kwargs, dataloader_kwargs):
    ds = CTCDataset(dataset, 'train', **dataset_kwargs)
    trainloader = DataLoader(ds, **dataloader_kwargs, shuffle=True)
    
    return trainloader



def load_valid(dataset, dataset_kwargs, dataloader_kwargs):
    ds = CTCDataset(dataset, 'valid', **dataset_kwargs)
    validloader = DataLoader(ds, **dataloader_kwargs, shuffle=False)
    
    return validloader



def load_test(dataset, dataset_kwargs, dataloader_kwargs):
    ds = CTCDataset(dataset, 'test', **dataset_kwargs)
    validloader = DataLoader(ds, **dataloader_kwargs, shuffle=False)
    
    return validloader



class CTCDataset(Dataset):
    
    def __init__(
        self, 
        dataset, 
        partition, 
        device='cpu', 
        im_size=None,
        load_limit=None
    ):
        root_dir = DATA_ROOT
        
        self.partition = partition
        self.device = device
        self.im_size = im_size
        
        if partition.lower() == 'train' or partition.lower() == 'valid':
            
            if partition.lower() == 'train':
                data_dir = os.path.join(root_dir, dataset, partition)
                
                imgs_dir = os.path.join(data_dir, '01')
                segs_gt_dir = os.path.join(data_dir, '01_GT', 'SEG')
                segs_st_dir = os.path.join(data_dir, '01_ST', 'SEG')
                
            else:
                data_dir = os.path.join(root_dir, dataset, 'train')
                
                imgs_dir = os.path.join(data_dir, '02')
                segs_gt_dir = os.path.join(data_dir, '02_GT', 'SEG')
                segs_st_dir = os.path.join(data_dir, '02_ST', 'SEG')
            
            imgs_paths = [os.path.join(imgs_dir, fp) for fp in sorted(os.listdir(imgs_dir))]
            segs_gt_st_paths = self.__get_segs_gt_st(segs_gt_dir, segs_st_dir)
            
            self.segs = self.__load_segs(segs_gt_st_paths, load_limit)
            
        elif partition.lower() == 'test':
            data_dir = os.path.join(root_dir, dataset, 'test')
            
            imgs_dir_1 = os.path.join(data_dir, '01')
            imgs_dir_2 = os.path.join(data_dir, '02')
            
            imgs_paths_1 = [os.path.join(imgs_dir_1, fp) for fp in sorted(os.listdir(imgs_dir_1))]
            imgs_paths_2 = [os.path.join(imgs_dir_2, fp) for fp in sorted(os.listdir(imgs_dir_2))]
            
            imgs_paths = [*imgs_paths_1, *imgs_paths_2]
            
        else:
            raise ValueError(f'parition "{partition}" is not valid')
        
        self.imgs = self.__load_imgs(imgs_paths, load_limit)
            
        
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        xs = self.imgs[idx]
        
        augment = self.partition == 'train'
        
        if self.partition != 'test':
            ys = self.segs[idx]
            
            if augment:
                xs, ys = random_hflip(xs, ys)
                xs, ys = random_vflip(xs, ys)
                xs, ys = random_rotate(xs, ys)
                xs, ys = random_roll(xs, ys)

            return xs, ys    
        
        else:
            return xs
        
        
        
    def __len__(self):
        return len(self.imgs)
        
        
        
    def __get_segs_gt_st(self, segs_gt_dir, segs_st_dir):
        segs_sts = sorted(os.listdir(segs_st_dir))
        segs_gts = sorted(os.listdir(segs_gt_dir))
        
        st_frames = [int(st[7:11]) for st in segs_sts]
        gt_frames = set([int(gt[7:11]) for gt in segs_gts])
        
        st_gts = []
        for st_frame in st_frames:
            if st_frame not in gt_frames: st_gts.append((segs_st_dir, segs_sts))
            else: st_gts.append((segs_gt_dir, segs_sts))
            
        segs_gt_st = [os.path.join(root, elems[i]) for i, (root, elems) in enumerate(st_gts)]
        
        return segs_gt_st
    
    
    
    def __load_imgs(self, img_paths, load_limit):   
        if load_limit is not None:
            img_paths = img_paths[:load_limit]
        
        imgs = []
        for img_path in tqdm(img_paths):
            img = sio.imread(img_path)
            
            if len(img.shape) == 3:
                if self.im_size is not None:
                    new_img = np.zeros(img.shape[-1], self.im_size, self.im_size)
                
                for i, slide in enumerate(np.transpose(img, (2, 0, 1))):
                    if self.im_size is not None:
                        slide = resize(slide, self.im_size, self.im_size)
                    
                    norm_slide = normalize(slide)
                    new_img[:, :, i] = norm_slide
                    
                img = new_img
                
            elif len(img.shape) == 2:
                if self.im_size is not None:
                    img = resize(img, self.im_size, self.im_size)
                    
                img = normalize(img)
            else:
                raise ValueError(f'invalid number of dimensions for image "{len(img.shape)}"')
            
            if len(img.shape) == 3:
                img = img[np.newaxis]
            elif len(img.shape) == 2:
                img = img[np.newaxis, :, :, np.newaxis]
                
            img = np.transpose(img, (3, 0, 2, 1))
            img = torch.tensor(img).float().to(self.device)
            
            imgs.append(img)
            
        imgs = torch.stack(imgs)
            
        return imgs
            
    
    
    def __load_segs(self, seg_paths, load_limit):        
        if load_limit is not None:
            seg_paths = seg_paths[:load_limit]
        
        segs = []
        for seg_path in tqdm(seg_paths):
            seg = sio.imread(seg_path)
            
            if 'GT' in seg_path:
                st_seg = sio.imread(seg_path[::-1].replace('TG', 'TS', 1)[::-1])

                if len(seg.shape) == 3:
                    st_seg = np.transpose(st_seg, (2, 0, 1))
                    gt_seg = np.transpose(seg, (2, 0, 1))
                    
                    for i, (gt_slide, st_slide) in enumerate(zip(gt_seg, st_seg)):
                        resolve_seg_conflicts(gt_slide, st_slide)
                        seg[:, :, i] = gt_slide
                        
                elif len(seg.shape) == 2:
                    resolve_seg_conflicts(seg, st_seg)
                else:
                    raise ValueError(f'invalid number of dimensions for segmentation mask "{len(seg.shape)}"')

            if len(seg.shape) == 3:
                if self.im_size is not None:
                    new_seg = np.zeros((self.im_size, self.im_size, seg.shape[-1]))
                
                for i, slide in enumerate(np.transpose(seg, (2, 0, 1))):
                    new_seg[:, :, i] = resize(slide, self.im_size, self.im_size)
                
                seg = new_seg
                seg = seg[np.newaxis]
            elif len(seg.shape) == 2:
                if self.im_size is not None: 
                    seg = resize(seg, self.im_size, self.im_size)
                    
                seg = seg[np.newaxis, :, :, np.newaxis]

            seg = np.transpose(seg, (3, 0, 2, 1))
            seg = torch.tensor(seg.astype(np.int16)).long().to(self.device)
                        
            segs.append(seg)
            
        segs = torch.stack(segs)
            
        return segs
    