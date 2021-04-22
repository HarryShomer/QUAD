import torch
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, gather_csr, scatter, segment_csr
from utils.utils_gcn import get_param, ccorr, rotate, softmax

from models.gnn_layer import *

### TODO: Fix!!!!
class CompGCNConvBasis(CompGCNConv):
    """
    CompGCN Conv Layer with basis for relations.

    Only used as first layer in multi-layer CompGCN
    """
    def __init__(self, prop_type, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
        super().__init__(prop_type, in_channels, out_channels, num_rels, act=act, params=params)

        self.num_bases  = self.p['num_bases']


    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.mm(self.rel_wt, self.rel_basis)
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_edges = edge_index.size(1) // 2
        num_quals = quals.size(1) // 2
        num_ent   = x.size(0)

        if not self.cache or self.in_norm is None:
            self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
            self.in_type,  self.out_type  = edge_type[:num_edges],   edge_type [num_edges:]

            self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
            self.loop_type  = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

            self.in_norm  = self.compute_norm(self.in_index,  num_ent)
            self.out_norm = self.compute_norm(self.out_index, num_ent)
        
        in_res   = self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm,   mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None,       mode='loop')
        out_res  = self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,  mode='out')
        out      = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

        if self.p.bias: 
            out = out + self.bias
        
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]









    def forward(self, x, edge_index, edge_type, rel_embed): 


        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type  = edge_type[:num_edges],   edge_type [num_edges:]

        self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

        self.in_norm     = self.compute_norm(self.in_index,  num_ent)
        self.out_norm    = self.compute_norm(self.out_index, num_ent)
        




    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):

        # B4 cat
        rel_embed = torch.mm(self.rel_wt, self.rel_basis)
        


