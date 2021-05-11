import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_gcn import get_param
# from .gnn_layer import *
from .old_gnn_layer import *


class HypRelEncoder(nn.Module):
    """
    Define the encoder
    """
    def __init__(self, graph_repr, config):
        super(self.__class__, self).__init__()

        self.device = config['DEVICE']
        self.act = torch.tanh if 'ACT' not in config['MODEL'] else config['MODEL']['ACT']

        self.emb_dim  = config['EMBEDDING_DIM']
        self.gcn_dim  = config['MODEL']['GCN_DIM']
        self.n_layer  = config['MODEL']['LAYERS']
        self.num_rel  = config['NUM_RELATIONS']
        self.num_ent  = config['NUM_ENTITIES']
        self.model_nm = config['MODEL_NAME'].lower()
        
        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)        
        self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)

        # Define Layers
        self.trip_conv1 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, params=config)
        self.qual_conv1 = CompGCNConv(self.gcn_dim, self.emb_dim, self.num_rel, act=self.act, params=config)
        self.both_conv1 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, params=config)

        self.trip_conv2 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, params=config)
        self.qual_conv2 = CompGCNConv(self.gcn_dim, self.emb_dim, self.num_rel, act=self.act, params=config)
        self.both_conv2 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act, params=config)

        self.trip_conv1 = self.trip_conv1.to(self.device)
        self.trip_conv2 = self.trip_conv2.to(self.device)

        self.qual_conv1 = self.qual_conv1.to(self.device)
        self.qual_conv2 = self.qual_conv2.to(self.device)

        self.both_conv1 = self.both_conv1.to(self.device)
        self.both_conv2 = self.both_conv2.to(self.device)

        self.drop1 = nn.Dropout(config['MODEL']['ENCODER_DROP_1'])
        self.drop2 = nn.Dropout(config['MODEL']['ENCODER_DROP_2'])




    def forward(self, ent_ix, rel_ix, quals_ix, ent_embs, rel_embs, aux_ent_embs=None, aux_rel_embs=None):
        """"
        Pass through encoder

        `aux` params are only used for both encoder

        :return:
        """
        # Trip/Qual and Both run in parallel 
        x1, r1 = self.trip_conv1("trip", ent_embs, self.edge_index, self.edge_type, rel_embs, quals=self.quals, aux_ents=aux_ent_embs, aux_rels=aux_rel_embs)
        x2, r2 = self.qual_conv1("qual", x1, self.edge_index, self.edge_type, r1, quals=self.quals, aux_ents=aux_ent_embs, aux_rels=aux_rel_embs)
        x3, r3 = self.both_conv1("both", x2, self.edge_index, self.edge_type, r2, quals=self.quals, aux_ents=aux_ent_embs, aux_rels=aux_rel_embs)

        x2 = self.drop1(x2)
        x3 = self.drop1(x3)

        # 2nd Layer in parallel
        x1, r1 = self.trip_conv2("trip", x2, self.edge_index, self.edge_type, r2, quals=self.quals, aux_ents=aux_ent_embs, aux_rels=aux_rel_embs)
        x2, r2 = self.qual_conv2("qual", x3, self.edge_index, self.edge_type, r3, quals=self.quals, aux_ents=aux_ent_embs, aux_rels=aux_rel_embs)
        x3, r3 = self.both_conv2("both", x2, self.edge_index, self.edge_type, r2, quals=self.quals, aux_ents=aux_ent_embs, aux_rels=aux_rel_embs)

        x1 = self.drop2(x1)
        x3 = self.drop2(x3)

        """
        We use:
            - Base Entities/Relations -> Output of trips
            - Qual Entities -> Output of both
        """

        rel_emb = torch.index_select(r1, 0, rel_ix)
        ent_emb = torch.index_select(x1, 0, ent_ix)

        # flatten quals
        quals_ents = quals_ix[:, 1::2].view(1,-1).squeeze(0)
        quals_rels = quals_ix[:, 0::2].view(1,-1).squeeze(0)

        qual_obj_emb = torch.index_select(x3, 0, quals_ents)
        qual_rel_emb = torch.index_select(r1, 0, quals_rels)

        qual_obj_emb = qual_obj_emb.view(ent_emb.shape[0], -1, ent_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        return ent_emb, rel_emb, qual_obj_emb, qual_rel_emb, x1, r1
