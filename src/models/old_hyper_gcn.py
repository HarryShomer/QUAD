import torch
import torch.nn as nn
import torch.nn.functional as F

# from .encoders import *
from .old_encoders import *
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
            raise ValueError("Invalid for for arg `emb_type`. Must be one of ['same', 'diff', 'base', 'project']")

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
        init_ent = self.ent_embs  
        init_rel = self.rel_embs

        # Same all way down
        e1_emb, r1_emb, qe1_emb, qr1_emb, x1, r1 = self.trip_encoder("trip", ent_ix, rel_ix, quals_ix, init_ent, init_rel)
        e2_emb, r2_emb, qe2_emb, qr2_emb, x2, r2 = self.qual_encoder("qual", ent_ix, rel_ix, quals_ix, x1, r1)
        e3_emb, r3_emb, qe3_emb, qr3_emb, x3, r3 = self.both_encoder("both", ent_ix, rel_ix, quals_ix, x2, r2)

        return self.decoder(e3_emb, r1_emb, qe3_emb, qr1_emb, x3, ents=ent_ix, quals=quals_ix)

        # if self.emb_type == "project":
        #     ent_trip_emb = torch.matmul(ent_both_emb, self.proj_out_matrix)
        #     ent_both_emb = torch.matmul(ent_both_emb, self.proj_in_matrix)


        # TODO: Sending trip for aux
        # e1_emb, r1_emb, qe1_emb, qr1_emb, x1, r1 = self.trip_encoder("trip", ent_ix, rel_ix, quals_ix, init_ent, init_rel)
        # e2_emb, r2_emb, qe2_emb, qr2_emb, x2, r2 = self.qual_encoder("multi-qual", ent_ix, rel_ix, quals_ix, x1, r1)
        # e3_emb, r3_emb, qe3_emb, qr3_emb, x3, r3 = self.both_encoder("multi-trip", ent_ix, rel_ix, quals_ix, x2, r2)

        # return self.decoder(e3_emb, r3_emb, qe3_emb, qr3_emb, x3, ents=ent_ix, quals=quals_ix)

