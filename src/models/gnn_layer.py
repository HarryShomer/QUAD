import torch
import numpy as np
import torch.nn.functional as F
from utils.utils_gcn import get_param, ccorr, rotate, softmax
from torch_scatter import scatter_add, scatter_min

from torch_geometric.nn import MessagePassing



class CompGCNConv(MessagePassing):
    """
    Standard Conv Layer for Compgcn
    """
    def __init__(self, in_channels, out_channels, num_rels, fact_encoder=None, act=lambda x:x, params=None):
        # target_to_source -> Edges that flow from x_i to x_j 
        super(self.__class__, self).__init__(flow='target_to_source', aggr='add')

        self.p            = params
        self.emb_dim      = params['EMBEDDING_DIM']
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_rels     = num_rels
        self.act          = act
        self.opn          = params['MODEL']['OPN']
        self.qual_comb    = params['MODEL']['QUAL_COMB']
        self.device       = None

        # Weight of both comp functions in 'both' aggregate
        self.alpha = params['ALPHA']

        # Three weight matrices for CompGCN 
        # In = Standard / Out = Inverse
        self.w_loop = get_param((in_channels, out_channels))
        self.w_in   = get_param((in_channels, out_channels))
        self.w_out  = get_param((in_channels, out_channels))

        # qual pairs
        self.w_q = get_param((in_channels, in_channels))

        # Weight matrix for relation update
        self.w_rel  = get_param((in_channels, out_channels))

        # TODO: Move out of here?
        # Rel embedding for loop triplets
        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p['MODEL']['GCN_DROP'])
        self.bn   = torch.nn.BatchNorm1d(out_channels)

        self.fact_encoder = fact_encoder

        

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


    def forward(self, prop_type, edge_index, edge_type, x, rel_embed, quals=None):
        """
        Forward by prop type for each type of aggregation
        """
        if self.device is None:
            self.device = edge_index.device

        num_ent   = x.size(0)
        num_quals = quals.size(1) // 2
        num_edges = edge_index.size(1) // 2

        # Add loop relation in!        
        rel_emb_all = torch.cat([rel_embed, self.loop_rel], dim=0)

        # 2nd half of triplets are inverse so split
        # in_{} = [col0, col1, ... col<num_edges - 1>]
        # out_{} = [col<num_edges>...]
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type,  out_type  = edge_type[:num_edges], edge_type [num_edges:]

        # Same as above but for qualifiers
        in_index_qual_ent, out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
        in_index_qual_rel, out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]

        # Refers to parent triplet qualifier belongs to
        quals_index_in, quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

        # Same for quals since same entity regardless....
        loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)   # Self edges between all the nodes
        loop_type = torch.full((num_ent,), rel_emb_all.size(0)-1, dtype=torch.long).to(self.device) # Last dim is for self-loop

        # Hack for fixing triplets for prop_type 'qual'
        if prop_type == "qual":
            # In -> (qv, s)
            in_index = torch.zeros(2, num_quals, dtype=torch.int64).to(self.device)
            in_index[0] = in_index_qual_ent
            in_index[1] = edge_index[0][quals_index_in]

            # Out -> (s, qv)
            out_index = torch.zeros(2, num_quals, dtype=torch.int64).to(self.device)
            out_index[0] = edge_index[0][quals_index_in]   # `in` is correct! for this and next line
            out_index[1] = in_index_qual_ent

            # The main edge_type here is qr
            in_type  = in_index_qual_rel
            out_type = out_index_qual_rel + self.num_rels

            # Account for triplet object. Comes from original non-inverse triplets
            trip_obj_in = edge_index[1][quals_index_in]
            trip_obj_out = edge_index[1][quals_index_out]

            # Account for relation in triplet obj
            trip_rel_in = trip_rel_out = in_type
        else:
            trip_obj_in = trip_obj_out = None
            trip_rel_in = trip_rel_out = None


        # Normalized Adj Matrices
        in_norm  = self.compute_norm(in_index,  num_ent)
        out_norm = self.compute_norm(out_index, num_ent)

        in_res = self.propagate(
                        in_index, 
                        x=x, 
                        edge_type=in_type,
                        rel_embed=rel_emb_all, 
                        edge_norm=in_norm,
                        mode='in',
                        ent_embed=x, 
                        qualifier_ent=in_index_qual_ent,
                        qualifier_rel=in_index_qual_rel,
                        qual_index=quals_index_in,
                        prop_type=prop_type,
                        trip_obj_ix=trip_obj_in,
                        trip_rel_ix=trip_rel_in
                    )


        if prop_type == "trip":
            out_res = self.propagate(
                            out_index, 
                            x=x, 
                            edge_type=out_type,
                            rel_embed=rel_emb_all, 
                            edge_norm=out_norm, 
                            mode='out',
                            ent_embed=x, 
                            qualifier_ent=out_index_qual_ent,
                            qualifier_rel=out_index_qual_rel,
                            qual_index=quals_index_out,
                            prop_type=prop_type,
                            trip_obj_ix=trip_obj_out,
                            trip_rel_ix=trip_rel_out,
                    )

        loop_res = self.propagate(
                        loop_index, 
                        x=x, 
                        edge_type=loop_type, 
                        rel_embed=rel_emb_all, 
                        edge_norm=None, 
                        mode='loop', 
                        prop_type=prop_type
                    )

        if prop_type == "trip":
            out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
        else:
            out = self.drop(in_res)*(1/2) + loop_res*(1/2)

        out = self.bn(out)

        # Ignoring the self loop inserted at the end since defined in this layer
        return self.act(out), torch.matmul(rel_emb_all, self.w_rel)[:-1]



    def message(self, x_i, x_j, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None, 
                qualifier_rel=None, qual_index=None, prop_type=None, trip_obj_ix=None, trip_rel_ix=None
        ):
        """
        Define message for MessagePassing class

        x_i = head emb
        x_j = tail emb

        1. Get appropriate weight matrix
        2. Type of aggregation depends on mode and prop_type
        3. Aggregation * Weight matrix -> output
        4. Pass output through norm when defined
        """
        comp_weight_matrix = getattr(self, 'w_{}'.format(mode))

        # Get relations embs used in sample
        base_rel_embs = torch.index_select(rel_embed, 0, edge_type)

        if mode == "loop":
            comp_agg = self.comp_func(x_j, base_rel_embs)
        elif prop_type == "trip":
            qual_ent_embed = ent_embed[qualifier_ent]
            qual_rel_embed = rel_embed[qualifier_rel]
            comp_agg = self.combine_trips(x_j, base_rel_embs, qual_ent_embed, qual_rel_embed, qual_index)
        elif prop_type == "qual":
            trip_rel_embed = rel_embed[trip_rel_ix]
            trip_obj_embed = ent_embed[trip_obj_ix]
            comp_agg = self.combine_quals(x_j, base_rel_embs, trip_obj_embed, trip_rel_embed)

        out = torch.mm(comp_agg, comp_weight_matrix)

        # Multiply by normalized adj matrix when applicable
        return out if edge_norm is None else out * edge_norm.view(-1, 1)


    def combine_trips(self, x_j, base_rel_embs, qual_ent_embs, qual_rel_embs, qual_to_trip_ix):
        """
        Combine the basic triplet info and sum for given head entity.

        Options based on value of self.qual_comb:

        1. 'none': phi(h_u, h_r)
        2. 'out':  a *  phi(h_u, h_r) + (1 - a) * h_q
        3. 'ent':  phi(a * h_u + (1 - a) * h_q, h_r)
        4. 'rel':  phi(h_u, a * h_r + (1 - a) * h_q)
        5. 'both': phi(a * h_u + (1 - a) * h_q, a * h_r + (1 - a) * h_q)
        """        
        # No inlusion of qual info
        if self.qual_comb == "none":
            return self.comp_func(x_j, base_rel_embs)
        
        # phi(qv, qr) and sum by base triplet
        qual_comp = self.comp_func(qual_ent_embs, qual_rel_embs)
        qual_coalesce = scatter_add(qual_comp, qual_to_trip_ix, dim=0, dim_size=base_rel_embs.shape[0])

        # h_q
        qualifier_emb = torch.mm(qual_coalesce, self.w_q)

        #######################
        # If trip has at least one qualifier -> 1 if True else 0
        # ones = torch.ones(qual_index.shape[0]).to(self.device)
        # trip_has_qual = scatter_min(ones, qual_index, dim=0, dim_size=base_rel_embs.shape[0])[0]

        # # Expand for emb dimension
        # trip_has_qual = trip_has_qual.repeat(self.emb_dim, 1).transpose(0, 1)
        # comp_agg = torch.where(trip_has_qual == 0, trip_agg, self.alpha * trip_agg + (1 - self.alpha) * qualifier_emb)
        #######################

        if self.qual_comb == "ent":
            x_j = self.alpha * x_j + (1 - self.alpha) * qualifier_emb 
            comp_agg = self.comp_func(x_j, base_rel_embs)

        elif self.qual_comb == "rel":
            base_rel_embs = self.alpha * base_rel_embs + (1 - self.alpha) * qualifier_emb 
            comp_agg = self.comp_func(x_j, base_rel_embs)

        elif self.qual_comb == "both":
            x_j = self.alpha * x_j + (1 - self.alpha) * qualifier_emb 
            base_rel_embs = self.alpha * base_rel_embs + (1 - self.alpha) * qualifier_emb 
            comp_agg = self.comp_func(x_j, base_rel_embs)
  
        elif self.qual_comb == "out":
            trip_agg = self.comp_func(x_j, base_rel_embs)
            comp_agg = self.alpha * trip_agg + (1 - self.alpha) * qualifier_emb 
        else:
            raise ValueError("Invalid value for argument --qual-comb")

        return comp_agg 


    def combine_quals(self, x_j, rel_embed, trip_obj_embed, trip_rel_embed):
        """
        For a given input combine the qualifier and triplet info
        
        In:
            x_j = s
            rel_embed = qr
            trip_rel_embed = r
            trip_obj_embed = o

        Out:
            x_j = qv
            rel_embed = qr^-1
            trip_rel_embed = r
            trip_obj_embed = o
        """        
        if self.p['FACT_ENCODER'] == "transformer":
            encoded_fact = self.fact_encoder(x_j, trip_obj_embed, trip_rel_embed)
        else:
            concat_fact = torch.cat((x_j, trip_obj_embed, trip_rel_embed), dim=1)
            encoded_fact = self.fact_encoder(concat_fact)

        return self.comp_func(encoded_fact, rel_embed)


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
        # Row = Head, Col = Tail for given set of triplets
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        
        # Num edges for each head entity
        # 0 for entities not in sample
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)   

        deg_inv = deg.pow(-0.5)                          # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col] # D^{-0.5}

        return norm
