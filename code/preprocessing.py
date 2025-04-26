import torch
import numpy as np
import scipy.sparse as sp

""" 
Symmetrically normalize adjacency matrix. This defines the feature propagation.
We will do the simple matrix operation. Here we do the part of the paper where 
S = D^{-1/2}(A+I)D^{-1/2}, specifically feature propagation on page 3.

Precondition: Adjacency matrix
Output: A "normalized" adjacency matrix with added self-loops
"""
def normalize_adj(adj):
    # Perform A+I
    adj = adj + torch.eye(adj.size(0))

    # degree vector
    deg = adj.sum(1)
    deg_inv_sqrt = deg.pow(-0.5)

    # Prevent divide by 0 errors
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    # dense
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
          
    return adj_norm.to_sparse()                        

"""Scale each node feature vector so its L1 norm equals 1."""
def row_normalize_features(x):
    row_sum = x.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return x / row_sum