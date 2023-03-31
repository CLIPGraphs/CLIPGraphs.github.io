import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch.nn import BatchNorm1d, BatchNorm2d, Dropout, LeakyReLU, Linear, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = GCNConv(input_dim, 256, normalize=True, cached=False)
        self.conv2 = GCNConv(256, 128, normalize=True, cached=False)
        self.conv3 = GCNConv(128, 64, normalize=True, cached=False)
        self.fc1 = Linear(64, 32)
        self.bn4 = LayerNorm(32)
        self.fc2 = Linear(32, 64)
        self.bn5 = LayerNorm(64)
        self.fc3 = Linear(64, self.output_dim)
        self.dropout = Dropout(0.4)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        '''
        x : Node features
        edge_index : (2,N) denoting edge connections
        edge_attr : Weight of each edge
        '''
        edge_attr = torch.clamp(edge_attr, min=0)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        x = self.dropout(x)

        x = self.lrelu(self.fc3(x))
        if torch.isnan(x).any():
            import pdb
            pdb.set_trace()
        return x
