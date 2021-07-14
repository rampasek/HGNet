"""
Adapted implementation from OGB's ogb/examples/graphproppred/mol/conv.py
https://github.com/snap-stanford/ogb
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class GINConvWParent(MessagePassing):
    def __init__(self, in_dim, emb_dim, edge_encoder=None):
        '''
        GIN convolution with a "parent" node
        '''

        super(GINConvWParent, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.eps2 = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = edge_encoder

    def forward(self, x, parent, edge_index, edge_attr=None):
        if self.edge_encoder is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = edge_attr
        if edge_embedding is None:
            edge_embedding = torch.zeros((edge_index.size(1), 1), device=edge_index.device)

        out = self.mlp((1 + self.eps) * x + \
                       (1 + self.eps2) * parent + \
                       self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConvWParent(MessagePassing):
    '''
    GCN convolution with a "parent" node
    '''

    def __init__(self, in_dim, emb_dim, edge_encoder=None):
        super(GCNConvWParent, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, in_dim)
        self.parent_emb = torch.nn.Embedding(1, in_dim)

        self.edge_encoder = edge_encoder

    def forward(self, x, parent, edge_index, edge_attr=None):
        if self.edge_encoder is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = edge_attr
        if edge_embedding is None:
            edge_embedding = torch.zeros((edge_index.size(1), 1), device=edge_index.device)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 2
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) \
              + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1) \
              + F.relu(parent + self.parent_emb.weight) * 1. / deg.view(-1, 1)
        out = self.linear(out)
        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
