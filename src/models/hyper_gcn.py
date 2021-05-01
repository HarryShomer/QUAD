import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import *
from .decoders import *
from utils.utils_gcn import get_param


class HypRelModel(nn.Module):

    def __init__(self, data, config):
        super(self.__class__, self).__init__()

        self.config = config
        self.device = config['DEVICE']
        self.num_rel = config['NUM_RELATIONS']
        self.num_ent = config['NUM_ENTITIES']
        self.emb_dim = config['EMBEDDING_DIM']
        self.emb_type = config['MODEL']['EMB_TYPE'].lower()

        if self.emb_type == "base":
            self.ent_bases   = get_param((self.num_ent, self.emb_dim))
            self.ent_in_wts  = get_param((self.num_ent, 1))
            self.ent_out_wts = get_param((self.num_ent, 1))

            self.rel_bases   = get_param((self.num_rel * 2, self.emb_dim))
            self.rel_in_wts  = get_param((self.num_rel * 2, 1))
            self.rel_out_wts = get_param((self.num_rel * 2, 1))

        elif self.emb_type == "diff":
            self.ent_in_embs = get_param((self.num_ent, self.emb_dim))
            self.ent_out_embs = get_param((self.num_ent, self.emb_dim))

            self.rel_in_embs = get_param((self.num_rel * 2, self.emb_dim))
            self.rel_out_embs = get_param((self.num_rel * 2, self.emb_dim))

        elif self.emb_type == "same" or self.emb_type == "project":
            self.ent_embs = get_param((self.num_ent, self.emb_dim))
            self.rel_embs = get_param((self.num_rel * 2, self.emb_dim))

            self.proj_in_matrix = get_param((self.emb_dim, self.emb_dim))
            self.proj_out_matrix = get_param((self.emb_dim, self.emb_dim))
        else:
            raise ValueError("Invalid for for arg `emb_type`. Must be one of ['same', 'diff', 'base']")

        self.trip_encoder = HypRelEncoder(data, config)
        self.trip_encoder.to(self.device)
        
        self.qual_encoder = HypRelEncoder(data, config)
        self.qual_encoder.to(self.device)

        self.both_encoder = HypRelEncoder(data, config)
        self.both_encoder.to(self.device)

        self.decoder = Transformer(config)
        self.decoder.to(self.device)

        self.loss = torch.nn.BCELoss()


    def get_ent_emb(self, pos):
        """
        pos = ['in', 'out']
        """
        if self.emb_type == "base":
            return self.ent_in_wts * self.ent_bases if pos =="in" else self.ent_out_wts * self.ent_bases

        if self.emb_type == "diff":
            return self.ent_in_embs if pos == "in" else self.ent_out_embs
        
        return self.ent_embs


    def get_rel_emb(self, pos):
        """
        pos = ['in', 'out']
        """
        if self.emb_type == "base":
            return self.rel_in_wts * self.rel_bases if pos =="in" else self.rel_out_wts * self.rel_bases

        if self.emb_type == "diff":
            return self.rel_in_embs if pos == "in" else self.rel_out_embs
        
        return self.rel_embs



    def forward(self, ent_ix, rel_ix, quals_ix):
        """
        1. Get embeddings (if bases or not)
        2. Forward each encoder
        3. Concat
        4. Decode final
        """
        init_ent = self.get_ent_emb("out")
        init_rel = self.get_rel_emb("out")

        """
        TODO
        ----

        Decoder:
          - Max_PAIRS
          - Concat operation
          - Not pass when h_v not found in Q?

        Encoder/Gnn-Layer:
            - Ensure aux_ent and aux_rel work correctly
        """

        # Same all way down
        ent_trip_emb, rel_trip_emb, _, _, x_trip, r_trip  = self.trip_encoder("trip", ent_ix, rel_ix, quals_ix, init_ent, init_rel)
        _, _, _, _, x_qual, r_qual = self.qual_encoder("qual", ent_ix, rel_ix, quals_ix, x_trip, r_trip)
        ent_both_emb, rel_both_emb, qual_ent_emb, qual_rel_emb, x, _ = self.both_encoder("both", ent_ix, rel_ix, quals_ix, x_qual, r_qual)

        if self.emb_type == "project":
            ent_trip_emb = torch.matmul(ent_both_emb, self.proj_out_matrix)
            ent_both_emb = torch.matmul(ent_both_emb, self.proj_in_matrix)

        # Scores
        return self.decoder(ent_trip_emb, ent_both_emb, rel_trip_emb, qual_ent_emb, qual_rel_emb, x, quals=quals_ix)

