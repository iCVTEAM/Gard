import seaborn
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt

import utils


class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat,  use_bn=False, mean=False, add_self=False):
        super(BatchedGraphSAGE, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn

        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).cuda()#.to()

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(h_k.size(1)).cuda()#.to()
            h_k = self.bn(h_k)
        if mask is not None:
            h_k = h_k * mask.unsqueeze(2).expand_as(h_k)

        return h_k

