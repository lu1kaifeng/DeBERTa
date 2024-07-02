import sys

sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from msg3d.graph.tools import k_adjacency, normalize_adjacency_matrix, k_adjacency_torch, \
    normalize_adjacency_matrix_torch, walk_path_torch
from msg3d.model.mlp import MLP
from msg3d.model.activation import activation_factory


class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales
        self.disentangled_agg = disentangled_agg
        self.in_channels = in_channels
        self.use_mask = use_mask
        self.projections = nn.ModuleList([ nn.Linear(self.in_channels,self.in_channels) for n in range(self.num_scales - 1)])
        self.mlp = nn.Linear(self.in_channels * (self.num_scales - 1),out_channels)

    def forward(self,A_binary,A_powers,A_lookup,A_last_edge, x):

        '''if self.disentangled_agg:
            A_powers = [k_adjacency_torch(A_binary, k,with_self=False) for k in range(self.num_scales)]
            A_powers = [normalize_adjacency_matrix_torch(g) for g in A_powers]
            A_powers = torch.stack(A_powers,dim=1)
        else:
            A_powers = [A_binary + torch.eye(len(A_binary)) for k in range(self.num_scales)]
            A_powers = [normalize_adjacency_matrix_torch(g) for g in A_powers]
            A_powers = [torch.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)'''
        #A_powers = torch.Tensor(A_powers)

        #N, C, T, V = x.shape
        #A_powers = A_powers.to(x.device)
        cumsum = torch.zeros([A_binary.shape[0],self.num_scales - 1,A_binary.shape[1],self.in_channels]).to(x.device)

        '''for i,b in enumerate(A_last_edge):
            for j,scale in enumerate(b):
                for fr,to in scale.nonzero():
                    weight = A_powers[i, j+1, fr, to]
                    last_edge = x[A_last_edge[i,j,fr,to] - 1, :]
                    cumsum[i,j,fr,:] += weight * last_edge
                pass'''

        for i,j,fr,to in A_last_edge.nonzero():
            weight = A_powers[i, j + 1, fr, to]
            last_edge = x[A_last_edge[i, j, fr, to] - 1, :]
            cumsum[i, j, fr, :] += weight * last_edge

        '''for k in range(1,self.num_scales):
            paths = walk_path_torch(A_lookup, k)
            for i, bat in enumerate(paths):
                for j, begin in enumerate(bat):
                    elem_cumsum = cumsum[i,k - 1,j,:]
                    for path in begin:
                        weight = A_powers[i,k,j,path[-1]]
                        last_edge = x[A_lookup[i, path[-2], path[-1]] - 1, :]
                        elem_cumsum += weight * last_edge'''
        #cumsum = torch.gather()

        for b in range(self.num_scales - 1):
            cumsum[:,b,:,:] = self.projections[b](cumsum[:,b,:,:])

        cumsum = cumsum.permute(0,2,1,3).flatten(start_dim=2,end_dim=3)
        cumsum = self.mlp(cumsum)

        '''A = A_powers.to(x.dtype)
        if self.use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(A_powers.shape)), -1e-6, 1e-6)
            A = A + A_res.to(x.dtype)'''

        #support = torch.einsum('nvu,nctu->nctv', A, x)
        #support = support.view(N, C, T, self.num_scales, V)
        #support = support.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)
        #out = self.mlp(support)
        return cumsum


'''if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16, 3, 30, 25))'''
