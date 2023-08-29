# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:25:24 2023

@author: Gavin
"""

import numpy as np

from scipy import ndimage

def find_closest_pairs(st_centroids, gt_centroids, threshold=10):
    st_centroids = np.array(st_centroids)
    gt_centroids = np.array(gt_centroids)
    
    pairwise_distances = np.sqrt(np.sum((st_centroids[:, np.newaxis] - gt_centroids) ** 2, axis=2))
    
    closest_indices = np.argwhere(pairwise_distances <= threshold)
    closest_pairs = np.array([(i, j) for i, j in closest_indices])
    
    unmatched_indices = set(
        range(st_centroids.shape[0])) - \
        set(closest_indices[:, 0]
    )
    
    unmatched = np.array(list(unmatched_indices))
    
    return closest_pairs, unmatched



def compute_centroids(lab, num_labels):
    centroids = []
    
    for label in range(1, num_labels + 1):
        labeled_region = lab == label
        centroid = ndimage.center_of_mass(labeled_region)
        centroids.append(centroid)
    
    return centroids