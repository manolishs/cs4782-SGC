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
    adj = adj + sp.eye(adj.shape[0])

    # Prepare matrix for conversion to PyTorch:
    adj = sp.coo_matrix(adj) # This is FROM paper itself

    # D: diagonal matri where each entry on the diagonal is equal to
    # the row-sum of the adjacency matrix
    row_sum = np.array(adj.sum(1))

    # Compute D^{-1/2}
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()

    # This handles division by 0 errors, make infinite entries 0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    # Get our D matrix
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # We use scipy cause its apparently optimized better, this is also from paper

    # Perform S = D^{-1/2}(A+I)D^{-1/2}
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
