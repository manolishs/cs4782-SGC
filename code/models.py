import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from preprocessing import row_normalize_features

    
"""
Recreation of a graph convolutional layer from the GCN paper
"""
class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot/Xavier uniform, GCN paper cites this use
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, adj_norm):
        # Perform X x W
        support = self.lin(x) 
        # Perform A x W
        return torch.spmm(adj_norm, support) if adj_norm.is_sparse \
               else adj_norm @ support   


"""
A Two-layer GCN.
"""
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

"""
SGC reimplementation, we assume that the features have been preprocessed 
with k-step graph propagation.
"""
class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias = True):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass, bias=bias)

    def forward(self, x):
        # Only do 1 linear layer as paper describes
        # REMINDER: DO SOFTMAX AFTER IN TRAINING
        return self.W(x)