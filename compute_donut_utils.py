import torch
import copy
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def fix_signs(components):
    for i in range(components.shape[0]):
        if components[i].mean() < 0:
            components[i] *= -1
    return components

def compute_phase(x, pca):
    x=pca.transform(x)
    return np.arctan2(x[:, 1], x[:, 0]) 

def compute_radius(x, pca):
    x=pca.transform(x)
    return np.sqrt(x[:, 1]**2+ x[:, 0]**2) 

def extract_rotation_angle(pca_x, n_neighbors=5):
    pca_data = copy.deepcopy(pca_x)[:, :3]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', ).fit(pca_data)
    distances, indices = nbrs.kneighbors(pca_data)

    tangent_vectors = np.zeros_like(pca_data)
    for i, idx in enumerate(indices):
        neighbors = pca_data[idx] - pca_data[i]
        tangent_vectors[i] = np.mean(neighbors, axis=0)

    pca_tangents = PCA(n_components=3)
   
    pca_tangents.fit(tangent_vectors)
    pca_tangents.components_ = fix_signs(pca_tangents.components_)
    rotation_matrix = pca_tangents.components_.T
    rotated_data = pca_data.dot(rotation_matrix)

    return rotated_data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # must be False for deterministic behavior

