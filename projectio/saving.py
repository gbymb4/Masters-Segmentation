# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:10:20 2023

@author: Gavin
"""

import os, json, yaml
import torch

import matplotlib.pyplot as plt

from torch import nn
from pconfig import OUT_DIR
from matplotlib.ticker import AutoMinorLocator

def plot_and_save_metric(train, valid, metric, fname):
    fig, ax = plt.subplots(figsize=(8, 6))

    epochs = list(range(1, len(train) + 1))

    ax.plot(epochs, train, label='Train', alpha=0.7)
    ax.plot(epochs, valid, label='Validation', alpha=0.7)   
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='major', length=4)
    ax.tick_params(which='minor', length=2, color='r')
    
    ax.legend()
    ax.grid(axis='y', c='white')
    
    ax.set_facecolor('whitesmoke')
    
    metric_name = (' '.join(metric.split('_'))).title()
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    
    plt.savefig(fname)
    plt.show()
    
    

def plot_and_save_visual(img, true, pred, post_pred, fname):
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    
    for ax, tensor, title in zip(
            axs, 
            [img, true, pred, post_pred], 
            ['Input', 'GT', 'Prediction', 'Postprocessed']
    ):
        ax.axis('off')
        ax.imshow(tensor.detach().cpu().numpy().transpose(1, 0), cmap='gray')
        ax.set_title(title, fontsize=24)
    
    plt.savefig(fname)
    plt.show()
    
    
    
def save_history_dict_and_model(
    dataset: str,
    model: nn.Module,
    id: int,
    config: dict,
    history: dict
) -> None:
    dataset = dataset.lower()
    model_name = type(model).__name__

    if not os.path.isdir(OUT_DIR): os.mkdir(OUT_DIR)

    save_root_dir = os.path.join(OUT_DIR, dataset)
    if not os.path.isdir(save_root_dir): os.mkdir(save_root_dir)
    
    save_model_dir = os.path.join(save_root_dir, model_name)
    if not os.path.isdir(save_model_dir): os.mkdir(save_model_dir)

    save_dir = os.path.join(save_model_dir, id)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    with open(os.path.join(save_dir, 'history.json'), 'w') as file:
        json.dump(history, file)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)
        
    for param in model.parameters():
        param.requires_grad = True

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))