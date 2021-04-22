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
        self.use_ent_bases = config['MODEL']['ENT_BASES']
        self.use_rel_bases = config['MODEL']['REL_BASES']


        self.ent_embs = get_param((self.num_ent, self.emb_dim))
        self.rel_embs = get_param((self.num_rel * 2, self.emb_dim))

        # if self.use_ent_bases:
        #     self.ent_bases   = get_param((self.num_ent, self.emb_dim))
        #     self.ent_in_wts  = get_param((self.num_ent, 1))
        #     self.ent_out_wts = get_param((self.num_ent, 1))
        # else:
        #     self.ent_in_embs = get_param((self.num_ent, self.emb_dim))
        #     self.ent_out_embs = get_param((self.num_ent, self.emb_dim))

        # # TODO: Additional bases?
        # if self.use_rel_bases:
        #     self.rel_bases   = get_param((self.num_rel * 2, self.emb_dim))
        #     self.rel_in_wts  = get_param((self.num_rel * 2, 1))
        #     self.rel_out_wts = get_param((self.num_rel * 2, 1))
        # else:
        #     self.rel_in_embs = get_param((self.num_rel * 2, self.emb_dim))
        #     self.rel_out_embs = get_param((self.num_rel * 2, self.emb_dim))


        self.trip_encoder = HypRelEncoder("trip", data, config)
        self.qual_encoder = HypRelEncoder("qual", data, config)
        self.both_encoder = HypRelEncoder("both", data, config)
        self.decoder = Transformer(config)

        self.trip_encoder.to(self.device)
        self.qual_encoder.to(self.device)
        self.both_encoder.to(self.device)
        self.decoder.to(self.device)

        self.loss = torch.nn.BCELoss()


    def forward(self, sub, rel, quals):
        """
        1. Get embeddings (if bases or not)
        2. Forward each encoder
        3. Concat
        4. Decode final
        """
        ## Create entity embs
        # if self.use_ent_bases:
        #     ent_in_embs  = self.ent_in_wts * self.ent_bases 
        #     ent_out_embs = self.ent_out_wts * self.ent_bases 
        # else:
        #     ent_in_embs = self.ent_in_embs
        #     ent_out_embs = self.ent_out_embs

        ## Create relation embs
        # if self.use_rel_bases:
        #     rel_in_embs  = self.rel_in_wts  * self.rel_bases
        #     rel_out_embs = self.rel_out_wts * self.rel_bases
        # else:
        #     rel_in_embs = self.rel_in_embs
        #     rel_out_embs = self.rel_out_embs 

        # Encode
        # sub_trip_emb, rel_trip_emb, _, _, _ = self.trip_encoder(sub, rel, quals, ent_out_embs, rel_out_embs)
        # _, _, _, _, _ = self.qual_encoder(sub, rel, quals, ent_in_embs, rel_in_embs)
        # sub_both_emb, rel_both_emb, qual_obj_emb, qual_rel_emb, x = self.both_encoder(sub, rel, quals, ent_in_embs, rel_in_embs)

        ## Scores
        # return self.decoder(sub_trip_emb, sub_both_emb, rel_trip_emb, qual_obj_emb, qual_rel_emb, x, quals=quals)

        sub_trip_emb, rel_trip_emb, _, _, _ = self.trip_encoder(sub, rel, quals, self.ent_embs, self.rel_embs)
        sub_qual_emb, rel_qual_emb, _, _, _ = self.qual_encoder(sub, rel, quals, self.ent_embs, self.rel_embs)
        sub_both_emb, rel_both_emb, qual_obj_emb, qual_rel_emb, x = self.both_encoder(sub, rel, quals, self.ent_embs, self.rel_embs)


        # Scores
        return self.decoder(sub_trip_emb, sub_both_emb, rel_trip_emb, qual_obj_emb, qual_rel_emb, x, quals=quals)
