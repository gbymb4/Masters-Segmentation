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
from preprocessing import (
    resolve_seg_conflicts, 
    normalize, 
    resize,
    get_markers,
    get_dummy_markers,
    tile_split,
    lcm_pad
)
from .augmentation import (
    random_hflip,
    random_vflip,
    random_rotate,
    random_roll
)

def load_train(dataset, dataset_type, dataset_kwargs, dataloader_kwargs):
    ds = dataset_type(dataset, 'train', **dataset_kwargs)
    trainloader = DataLoader(ds, **dataloader_kwargs, shuffle=True)
    
    return trainloader



def load_valid(dataset, dataset_type, dataset_kwargs, dataloader_kwargs):
    ds = dataset_type(dataset, 'valid', **dataset_kwargs)
    validloader = DataLoader(ds, **dataloader_kwargs, shuffle=False)
    
    return validloader



def load_test(dataset, dataset_type, dataset_kwargs, dataloader_kwargs):
    ds = dataset_type(dataset, 'test', **dataset_kwargs)
    validloader = DataLoader(ds, **dataloader_kwargs, shuffle=False)
    
    return validloader



class CTCDataset(Dataset):
    
    def __init__(
        self, 
        dataset, 
        partition, 
        device='cpu', 
        im_size=None,
        tile_size=None,
        load_limit=None
    ):
        root_dir = DATA_ROOT
        
        self.partition = partition
        self.device = device
        self.im_size = im_size
        self.tile_size = tile_size
        
        if partition.lower() == 'train' or partition.lower() == 'valid' or partition.lower() == 'test':
            
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
            
            if partition.lower() == 'valid':
                imgs_paths = [elem for i, elem in enumerate(imgs_paths) if i % 2 == 0]
                segs_gt_st_paths = [elem for i, elem in enumerate(segs_gt_st_paths) if i % 2 == 0]
            
            elif partition.lower() == 'test':
                imgs_paths = [elem for i, elem in enumerate(imgs_paths) if i % 2 == 1]
                segs_gt_st_paths = [elem for i, elem in enumerate(segs_gt_st_paths) if i % 2 == 1]            
            
            self.segs = self.__load_segs(segs_gt_st_paths, load_limit)
            
        elif partition.lower() == 'eval':
            data_dir = os.path.join(root_dir, dataset, 'eval')
            
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
        
        if self.partition != 'eval':
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
        
        def get_id(fp):
            return int(fp[7:].split('.')[0])
        
        st_frames = [get_id(st) for st in segs_sts]
        gt_frames = set([get_id(gt) for gt in segs_gts])
        
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
                
            img = np.transpose(img, (3, 2, 1, 0))
            img = np.concatenate([tile_split(
                lcm_pad(i, self.tile_size), 
                self.tile_size
            ) for i in img], axis=0)
            
            img = np.transpose(img, (0, 3, 2, 1))
            img = torch.tensor(img).float().to(self.device)
            
            imgs.append(img)
            
        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.unsqueeze(dim=1)
        
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
            
            seg = np.transpose(seg, (3, 2, 1, 0))
            seg = np.concatenate([tile_split(
                lcm_pad(i, self.tile_size),
                self.tile_size
            ) for i in seg], axis=0)
            
            seg = np.transpose(seg, (0, 3, 2, 1))
            
            markers = get_markers(seg)
            
            seg = np.stack((seg, markers), axis=1)
            seg = torch.tensor(seg.astype(np.int16)).long().to(self.device)

            segs.append(seg)
            
        segs = torch.cat(segs, dim=0)
        
        return segs



class DRIVEDataset(Dataset):
    
    def __init__(
        self, 
        _, 
        partition, 
        device='cpu', 
        tile_size=48,
        train_frac=0.9,
        valid_frac=0.05,
        test_frac=0.05,
        load_limit=None
    ):
        assert round(train_frac + valid_frac + test_frac, 3) == 1.0
        
        root_dir = DATA_ROOT
        
        self.partition = partition
        self.device = device
        self.tile_size = tile_size
        
        data_root = os.path.join(root_dir, 'DRIVE', 'training')
        seg_root = os.path.join(data_root, '1st_manual')
        img_root = os.path.join(data_root, 'images')
        
        img_fps = [os.path.join(img_root, fp)for fp in sorted(
            os.listdir(img_root),
            key=lambda x: int(x[:-13])
        )]
        seg_fps = [os.path.join(seg_root, fp)for fp in sorted(
            os.listdir(seg_root),
            key=lambda x: int(x[:-13])
        )]
        
        num_imgs = len(img_fps)
        
        if self.partition == 'train':
            start_idx = 0
            end_idx = int(num_imgs * train_frac)
        
        elif self.partition == 'valid':
            start_idx = int(num_imgs * train_frac)
            end_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            
        elif self.partition == 'test':
            start_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            end_idx = num_imgs
        
        elif self.partition == 'eval':
            raise ValueError('"eval" partition not supported for STAREDataset.')
        
        filtered_img_fps = img_fps[start_idx : end_idx]
        filtered_seg_fps = seg_fps[start_idx : end_idx]
        
        if load_limit is not None:
            filtered_img_fps = filtered_img_fps[:load_limit]
            filtered_seg_fps = filtered_seg_fps[:load_limit]
        
        self.imgs, self.segs = self.__load_ppms(
            filtered_img_fps,
            filtered_seg_fps
        )
        
        
        
    def __load_ppms(self, img_fps, seg_fps):
        imgs, segs = [], []
        
        for img_fp, seg_fp in zip(img_fps, seg_fps):
            img = sio.imread(img_fp) / 255
            seg = sio.imread(seg_fp)
            
            seg = seg > 0
            seg = seg[..., np.newaxis]
            seg = lcm_pad(seg, self.tile_size)
            seg = tile_split(seg, self.tile_size)
            seg = seg.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            seg = get_dummy_markers(seg)
            seg = torch.tensor(seg).long().to(self.device)
            
            segs.append(seg)

            img = lcm_pad(img, self.tile_size)
            img = tile_split(img, self.tile_size)
            img = img.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            img = torch.tensor(img).float().to(self.device)
            
            imgs.append(img)
        
        imgs = torch.cat(imgs, 0)
        segs = torch.cat(segs, 0)
        
        return imgs, segs
    
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        xs = self.imgs[idx]
        
        augment = self.partition == 'train'
        
        if self.partition != 'eval':
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



class STAREDataset(Dataset):
    
    def __init__(
        self, 
        _, 
        partition, 
        device='cpu', 
        tile_size=48,
        train_frac=0.9,
        valid_frac=0.05,
        test_frac=0.05,
        load_limit=None
    ):
        assert round(train_frac + valid_frac + test_frac, 3) == 1.0
        
        root_dir = DATA_ROOT
        
        self.partition = partition
        self.device = device
        self.tile_size = tile_size
        
        data_root = os.path.join(root_dir, 'STARE')
        ah_root = os.path.join(data_root, 'annotations', 'ah')
        vk_root = os.path.join(data_root, 'annotations', 'vk')
        
        img_fps = [os.path.join(data_root, 'imgs', fp)for fp in sorted(
            os.listdir(os.path.join(data_root, 'imgs')),
            key=lambda x: int(x[2:-4])
        )]
        ah_fps = [os.path.join(ah_root, fp)for fp in sorted(
            os.listdir(ah_root),
            key=lambda x: int(x[2:-7])
        )]
        vk_fps = [os.path.join(vk_root, fp)for fp in sorted(
            os.listdir(vk_root),
            key=lambda x: int(x[2:-7])
        )]
        
        num_imgs = len(img_fps)
        
        if self.partition == 'train':
            start_idx = 0
            end_idx = int(num_imgs * train_frac)
        
        elif self.partition == 'valid':
            start_idx = int(num_imgs * train_frac)
            end_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            
        elif self.partition == 'test':
            start_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            end_idx = num_imgs
        
        elif self.partition == 'eval':
            raise ValueError('"eval" partition not supported for STAREDataset.')
        
        filtered_img_fps = img_fps[start_idx : end_idx]
        filtered_ah_fps = ah_fps[start_idx : end_idx]
        filtered_vk_fps = vk_fps[start_idx : end_idx]
        
        if load_limit is not None:
            filtered_img_fps = filtered_img_fps[:load_limit]
            filtered_ah_fps = filtered_ah_fps[:load_limit]
            filtered_vk_fps = filtered_vk_fps[:load_limit]
        
        self.imgs, self.segs = self.__load_ppms(
            filtered_img_fps,
            filtered_ah_fps,
            filtered_vk_fps
        )
        
        
        
    def __load_ppms(self, img_fps, ah_fps, vk_fps):
        imgs, segs = [], []
        
        for img_fp, ah_fp, vk_fp in zip(img_fps, ah_fps, vk_fps):
            img = np.array(Image.open(img_fp).convert('RGB')) / 255
            ah = np.array(Image.open(ah_fp))
            vk = np.array(Image.open(vk_fp))
            
            seg = ((ah > 0) | (vk > 0))
            seg = seg[..., np.newaxis]
            seg = lcm_pad(seg, self.tile_size)
            seg = tile_split(seg, self.tile_size)
            seg = seg.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            seg = get_dummy_markers(seg)
            seg = torch.tensor(seg).long().to(self.device)
            
            segs.append(seg)

            img = lcm_pad(img, lcm=self.tile_size)
            img = tile_split(img, self.tile_size)
            img = img.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            img = torch.tensor(img).float().to(self.device)
            
            imgs.append(img)
        
        imgs = torch.cat(imgs, 0)
        segs = torch.cat(segs, 0)
        
        return imgs, segs
    
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        xs = self.imgs[idx]
        
        augment = self.partition == 'train'
        
        if self.partition != 'eval':
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
        
    
    
class IOSTARDataset(Dataset):
    
    def __init__(
        self, 
        _, 
        partition, 
        device='cpu', 
        tile_size=48,
        train_frac=0.9,
        valid_frac=0.05,
        test_frac=0.05,
        load_limit=None
    ):
        assert round(train_frac + valid_frac + test_frac, 3) == 1.0
        
        root_dir = DATA_ROOT
        
        self.partition = partition
        self.device = device
        self.tile_size = tile_size
        
        data_root = os.path.join(root_dir, 'IOSTAR')
        seg_root = os.path.join(data_root, 'GT')
        img_root = os.path.join(data_root, 'image')
        
        img_fps = [os.path.join(img_root, fp)for fp in sorted(
            os.listdir(img_root),
            key=lambda x: int(x[5:-8])
        )]
        seg_fps = [os.path.join(seg_root, fp)for fp in sorted(
            os.listdir(seg_root),
            key=lambda x: int(x[5:-11])
        )]
        
        num_imgs = len(img_fps)
        
        if self.partition == 'train':
            start_idx = 0
            end_idx = int(num_imgs * train_frac)
        
        elif self.partition == 'valid':
            start_idx = int(num_imgs * train_frac)
            end_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            
        elif self.partition == 'test':
            start_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            end_idx = num_imgs
        
        elif self.partition == 'eval':
            raise ValueError('"eval" partition not supported for STAREDataset.')
        
        filtered_img_fps = img_fps[start_idx : end_idx]
        filtered_seg_fps = seg_fps[start_idx : end_idx]
        
        if load_limit is not None:
            filtered_img_fps = filtered_img_fps[:load_limit]
            filtered_seg_fps = filtered_seg_fps[:load_limit]
        
        self.imgs, self.segs = self.__load_ppms(
            filtered_img_fps,
            filtered_seg_fps
        )
        
        
        
    def __load_ppms(self, img_fps, seg_fps):
        imgs, segs = [], []
        
        for img_fp, seg_fp in zip(img_fps, seg_fps):
            img = np.array(Image.open(img_fp).convert('RGB')) / 255
            seg = np.array(Image.open(seg_fp))
            
            seg = seg > 0
            seg = seg[..., np.newaxis]
            seg = lcm_pad(seg, lcm=self.tile_size)
            seg = tile_split(seg, self.tile_size)
            seg = seg.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            seg = get_dummy_markers(seg)
            seg = torch.tensor(seg).long().to(self.device)
            
            segs.append(seg)

            img = lcm_pad(img, lcm=self.tile_size)
            img = tile_split(img, self.tile_size)
            img = img.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            img = torch.tensor(img).float().to(self.device)
            
            imgs.append(img)
        
        imgs = torch.cat(imgs, 0)
        segs = torch.cat(segs, 0)
        
        return imgs, segs
    
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        xs = self.imgs[idx]
        
        augment = self.partition == 'train'
        
        if self.partition != 'eval':
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
    
    
    
class HRFDataset(Dataset):
    
    def __init__(
        self, 
        _, 
        partition, 
        device='cpu', 
        tile_size=48,
        train_frac=0.9,
        valid_frac=0.05,
        test_frac=0.05,
        load_limit=None
    ):
        assert round(train_frac + valid_frac + test_frac, 3) == 1.0
        
        root_dir = DATA_ROOT
        
        self.partition = partition
        self.device = device
        self.tile_size = tile_size
        
        data_root = os.path.join(root_dir, 'HRF')
        seg_root = os.path.join(data_root, 'manual1')
        img_root = os.path.join(data_root, 'images')
        
        img_fps = [os.path.join(img_root, fp)for fp in sorted(
            os.listdir(img_root)
        )]
        seg_fps = [os.path.join(seg_root, fp)for fp in sorted(
            os.listdir(seg_root)
        )]
        
        num_imgs = len(img_fps)
        
        if self.partition == 'train':
            start_idx = 0
            end_idx = int(num_imgs * train_frac)
        
        elif self.partition == 'valid':
            start_idx = int(num_imgs * train_frac)
            end_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            
        elif self.partition == 'test':
            start_idx = int(num_imgs * round(train_frac + valid_frac, 3))
            end_idx = num_imgs
        
        elif self.partition == 'eval':
            raise ValueError('"eval" partition not supported for STAREDataset.')
        
        filtered_img_fps = img_fps[start_idx : end_idx]
        filtered_seg_fps = seg_fps[start_idx : end_idx]
        
        if load_limit is not None:
            filtered_img_fps = filtered_img_fps[:load_limit]
            filtered_seg_fps = filtered_seg_fps[:load_limit]
        
        self.imgs, self.segs = self.__load_ppms(
            filtered_img_fps,
            filtered_seg_fps
        )
        
        
        
    def __load_ppms(self, img_fps, seg_fps):
        imgs, segs = [], []
        
        for img_fp, seg_fp in zip(img_fps, seg_fps):
            img = sio.imread(img_fp) / 255
            seg = sio.imread(seg_fp)
            
            seg = seg > 0
            seg = seg[..., np.newaxis]
            seg = lcm_pad(seg, lcm=self.tile_size)
            seg = tile_split(seg, self.tile_size)
            seg = seg.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            seg = get_dummy_markers(seg)
            seg = torch.tensor(seg).long().to(self.device)
            
            segs.append(seg)

            img = lcm_pad(img, lcm=self.tile_size)
            img = tile_split(img, self.tile_size)
            img = img.transpose(0, 3, 2, 1)[:, :, np.newaxis, ...]
            img = torch.tensor(img).float().to(self.device)
            
            imgs.append(img)
        
        imgs = torch.cat(imgs, 0)
        segs = torch.cat(segs, 0)
        
        return imgs, segs
    
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        xs = self.imgs[idx]
        
        augment = self.partition == 'train'
        
        if self.partition != 'eval':
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
    
    