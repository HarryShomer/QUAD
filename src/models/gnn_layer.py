import torch
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, gather_csr, scatter, segment_csr
from utils.utils_gcn import get_param, ccorr, rotate, softmax

from torch_geometric.nn import MessagePassing
# from .message_passing import *


class CompGCNConv(MessagePassing):
    """
    Standard Conv Layer for Compgcn
    """
    def __init__(self, prop_type, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
        # target_to_source -> Edges that flow from x_i to x_j 
        super(self.__class__, self).__init__(flow='target_to_source', aggr='add')

        self.prop_type    = prop_type
        self.p            = params
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_rels     = num_rels
        self.act          = act
        self.opn          = params['MODEL']['OPN']
        self.device       = None

        # Weight of both comp functions in 'both' aggregate
        self.alpha = params['ALPHA']

        # Three weight matrices for CompGCN 
        # In = Standard / Out = Inverse
        self.w_loop = get_param((in_channels, out_channels))
        self.w_in   = get_param((in_channels, out_channels))
        self.w_out  = get_param((in_channels, out_channels))

        # Weight matrix for relation update
        self.w_rel  = get_param((in_channels, out_channels))

        # TODO: Move out of here?
        # Rel embedding for loop triplets
        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p['MODEL']['GCN_DROP'])
        self.bn   = torch.nn.BatchNorm1d(out_channels)

        if self.p['MODEL']['BIAS']: 
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))


    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


    def forward(self, x, edge_index, edge_type, rel_embed, quals=None): 
        """
        """
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_quals = quals.size(1) // 2
        num_ent   = x.size(0)

        # 2nd half of triplets are inverse so split
        # in_{} = [col1, col2, ... col<num_edges>]
        # out_{} = [col<num_edges + 1>...]
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type,  out_type  = edge_type[:num_edges], edge_type [num_edges:]

        # Same as above but for qualifiers
        in_index_qual_ent, out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
        in_index_qual_rel, out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]

        # Refers to parent triplet qualifier belongs to
        # 1st is index of triplet in edge_index
        # 2nd is entity head for triplet that it belongs to
        quals_index_in, quals_index_out = quals[2, :num_quals], quals[2, num_quals:]
        quals_head_index_in, quals_head_index_out = edge_index[0][quals_index_in], edge_index[0][quals_index_out]

        # Same for quals since same entity regardless....
        loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)   # Self edges between all the nodes
        loop_type = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)   # Last dim is for self-loop

        in_norm   = self.compute_norm(in_index,  num_ent)
        out_norm  = self.compute_norm(out_index, num_ent)

        in_res = self.propagate(in_index, x=x, edge_type=in_type,
                                rel_embed=rel_embed, edge_norm=in_norm, mode='in',
                                ent_embed=x, qualifier_ent=in_index_qual_ent,
                                qualifier_rel=in_index_qual_rel,
                                qual_index=quals_index_in, head_index=quals_head_index_in,
                                prop_type=self.prop_type)

        out_res = self.propagate(out_index, x=x, edge_type=out_type,
                                 rel_embed=rel_embed, edge_norm=out_norm, mode='out',
                                 ent_embed=x, qualifier_ent=out_index_qual_ent,
                                 qualifier_rel=out_index_qual_rel,
                                 qual_index=quals_index_out, head_index=quals_head_index_out,
                                 prop_type=self.prop_type)

        loop_res = self.propagate(loop_index, x=x, edge_type=loop_type, rel_embed=rel_embed, 
                                  edge_norm=None, mode='loop', prop_type=self.prop_type)

        out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
        
        if self.p['MODEL']['BIAS']: 
            out = out + self.bias
        
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]      # Ignoring the self loop inserted at the end


    def message(self, x_i, x_j, edge_type, rel_embed, edge_norm, mode, ent_embed=None, 
                qualifier_ent=None, qualifier_rel=None, qual_index=None, head_index=None, prop_type=None):
        """
        Define message for MessagePassing class

        1. Get appropriate weight matrix
        2. Type of aggregation depends on mode and prop_type
        3. Aggregation x Weight matrix -> output
        4. Pass output through norm when defined
        """
        comp_weight_matrix = getattr(self, 'w_{}'.format(mode))

        # Get relations embs used in sample
        # For loop edge_type is fully pointing to the loop relation
        rel_sub_embs = torch.index_select(rel_embed, 0, edge_type)

        # 1. Loop...basically same as trip but no scatter (since only one edge per entity)
        # 2. Rel & tail to head (v <- u, r)
        # 3. Quals to head (v <- qv, qr)
        # 4. Main triplet and qual to qual (qv <- v, u, r, qr)
        if mode == "loop":
            comp_agg = self.comp_func(x_j, rel_sub_embs)
        elif prop_type == "trip":
            comp_agg = self.combine_trips(x_j, rel_sub_embs, ent_embed, head_index)
        elif prop_type == "qual":
            comp_agg = self.combine_quals(ent_embed, rel_sub_embs, qualifier_ent, qualifier_rel, qual_index)
        elif prop_type == "both":
            comp_agg = self.combine_quals_trips(x_i, x_j, ent_embed, rel_sub_embs, qualifier_ent, qualifier_rel, qual_index, head_index)

        out = torch.mm(comp_agg, comp_weight_matrix)

        # Multiply by normalized adj matrix when applicable
        return out if edge_norm is None else out * edge_norm.view(-1, 1)



    def combine_trips(self, x_j, rel_sub_embs, ent_embed, head_index):
        """
        Combine the basic triplet info and sum for given head entity
        """
        comp_func_out = self.comp_func(x_j, rel_sub_embs)
        return comp_func_out


    def combine_quals_trips(self, x_i, x_j, ent_embed, rel_embed, qualifier_ent, qualifier_rel, qual_index, quals_head_index):
        """
        For a given input combine the qualifier and triplet info
        """
        num_edges = rel_embed.shape[0]

        # Retrieve embeddings for quals
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        # Corresponding triplets for quals
        x_i = x_i[qual_index]
        x_j = x_j[qual_index]
        rel_embed = rel_embed[qual_index]
        
        # Comp func for trips
        comp_trip_agg  = self.comp_func(x_j, rel_embed)

        # Qual agg here is h_v and h_qr
        qual_embeddings = self.comp_func(x_i, qualifier_emb_rel)

        qual_trip_sum = self.alpha * comp_trip_agg + (1 - self.alpha) * qual_embeddings

        return scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)


    def combine_quals(self, ent_embed, rel_embed, qualifier_ent, qualifier_rel, quals_index):
        """
        For a given input combine the qualifier info
        """
        # Retrieve embeddings for quals
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        qual_embeddings = self.comp_func(qualifier_emb_ent, qualifier_emb_rel)

        # Add up qual pairs that refer to same triplet 
        coalesced_quals = scatter_add(qual_embeddings, quals_index, dim=0, dim_size=rel_embed.shape[0])

        return coalesced_quals


    def comp_func(self, ent_embed, rel_embed):
        """
        phi_r
        """
        if self.opn == 'corr':  
            trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub':   
            trans_embed  = ent_embed - rel_embed
        elif self.opn == 'mult':  
            trans_embed  = ent_embed * rel_embed
        elif self.opn == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else: 
            raise NotImplementedError

        return trans_embed



    def update(self, aggr_out):
        return aggr_out


    def compute_norm(self, edge_index, num_ent):
        """
        Computes the normalized adj matrix
        """
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg  = scatter_add( edge_weight, row, dim=0, dim_size=num_ent)   # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)                         # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]         # D^{-0.5}

        return norm



    def qual_inverse_edges(self, edge_index, edge_type, quals):
        """
        Create inverse edges for prop_type 'qual'
        """
        num_quals = quals.size(1) // 2
        num_edges = edge_index.size(1) // 2

        inverse_edges = edge_type[:num_edges] + self.num_rels

        # Same as above but for qualifiers
        index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
        index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
        
        #tails = 

        # Flip head to tail
        edge_index[1, :len(raw)] = edge_index[0, :len(raw)]


        # 2nd half of triplets are inverse so split
        # in_{} = [col1, col2, ... col<num_edges>]
        # out_{} = [col<num_edges + 1>...]
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type,  out_type  = edge_type[:num_edges], edge_type [num_edges:]


        # Refers to parent triplet qualifier belongs to
        # 1st is index of triplet in edge_index
        # 2nd is entity head for triplet that it belongs to
        quals_index_in, quals_index_out = quals[2, :num_quals], quals[2, num_quals:]





# ### TODO: Fix!!!!
# class CompGCNConvBasis(CompGCNConv):
#     """
#     CompGCN Conv Layer with basis for relations.

#     Only used as first layer in multi-layer CompGCN
#     """
#     def __init__(self, in_channels, out_channels, num_rels, num_bases, act=lambda x:x, cache=True, params=None):
#         super().__init__(in_channels, out_channels, num_rels, act=act, params=params)

#         self.cache      = cache
#         self.num_bases  = num_bases
#         self.in_norm, self.out_norm = None, None
#         self.in_index, self.out_index = None, None
#         self.in_type, self.out_type = None, None
#         self.loop_index, self.loop_type = None, None


#     def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
#         if self.device is None:
#             self.device = edge_index.device

#         rel_embed = torch.mm(self.rel_wt, self.rel_basis)
#         rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

#         num_edges = edge_index.size(1) // 2
#         num_quals = quals.size(1) // 2
#         num_ent   = x.size(0)

#         if not self.cache or self.in_norm is None:
#             self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
#             self.in_type,  self.out_type  = edge_type[:num_edges],   edge_type [num_edges:]

#             self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
#             self.loop_type  = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

#             self.in_norm  = self.compute_norm(self.in_index,  num_ent)
#             self.out_norm = self.compute_norm(self.out_index, num_ent)
        
#         in_res   = self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm,   mode='in')
#         loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None,       mode='loop')
#         out_res  = self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,  mode='out')
#         out      = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

#         if self.p.bias: 
#             out = out + self.bias
        
#         out = self.bn(out)

#         return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]





