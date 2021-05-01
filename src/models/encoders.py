import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnn_layer import *
from utils.utils_gcn import get_param


class HypRelEncoder(nn.Module):
    """
    Define the encoder
    """
    def __init__(self, graph_repr, config):
        super(self.__class__, self).__init__()

        self.device = config['DEVICE']
        self.act = torch.tanh if 'ACT' not in config['MODEL'] else config['MODEL']['ACT']

        self.emb_dim   = config['EMBEDDING_DIM']
        self.gcn_dim   = config['MODEL']['GCN_DIM']
        self.n_layer   = config['MODEL']['LAYERS']
        self.num_rel   = config['NUM_RELATIONS']
        self.num_ent   = config['NUM_ENTITIES']
        self.model_nm  = config['MODEL_NAME'].lower()
        
        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)        
        self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)

        # Define Layers
        self.conv1 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, params=config)
        self.conv2 = CompGCNConv(self.gcn_dim, self.emb_dim, self.num_rel, act=self.act, params=config)

        self.drop1 = nn.Dropout(config['MODEL']['ENCODER_DROP_1'])
        self.drop2 = nn.Dropout(config['MODEL']['ENCODER_DROP_2'])

        self.conv1 = self.conv1.to(self.device)
        self.conv2 = self.conv2.to(self.device)

        self.register_parameter('bias', nn.Parameter(torch.zeros(self.num_ent)))



    def forward(self, prop_type, ent_ix, rel_ix, quals_ix, ent_embs, rel_embs, aux_ent_embs=None, aux_rel_embs=None):
        """"
        Pass through encoder

        `aux` params are only used for both encoder

        :return:
        """
        x, r = self.conv1(
                    prop_type=prop_type,
                    x=ent_embs, 
                    edge_index=self.edge_index, 
                    edge_type=self.edge_type, 
                    rel_embed=rel_embs, 
                    quals=self.quals,
                )
        x = self.drop1(x)

        if self.n_layer == 2:
            x, r = self.conv2(
                        prop_type=prop_type,
                        x=x, 
                        edge_index=self.edge_index, 
                        edge_type=self.edge_type, 
                        rel_embed=r, 
                        quals=self.quals,
                    )
            x = self.drop2(x) 

        sub_emb = torch.index_select(x, 0, ent_ix)
        rel_emb = torch.index_select(r, 0, rel_ix)

        # flatten quals
        quals_ents = quals_ix[:, 1::2].view(1,-1).squeeze(0)
        quals_rels = quals_ix[:, 0::2].view(1,-1).squeeze(0)

        qual_obj_emb = torch.index_select(x, 0, quals_ents)
        qual_rel_emb = torch.index_select(r, 0, quals_rels)

        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x, r
