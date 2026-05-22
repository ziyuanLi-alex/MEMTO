import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kmeans_pytorch import kmeans
import time


def to_device(x):
    """Move tensor to CUDA if available."""
    if torch.cuda.is_available():
        return x.cuda()
    return x


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    # sckit-learn xxxx (cuda problem)
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers