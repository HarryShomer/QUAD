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

        # self.ent_trip_embs = get_param((self.num_ent, self.emb_dim)).to(self.device)
        # self.ent_qual_embs = get_param((self.num_ent, self.emb_dim)).to(self.device)

        # # *2 for inverse edges
        # self.rel_trip_embs = get_param((self.num_rel * 2, self.emb_dim)).to(self.device)
        # self.rel_qual_embs = get_param((self.num_rel * 2, self.emb_dim)).to(self.device)

        # For combining trip and qual embeddings
        # This is weight given to trip
        # self.beta = self.config['BETA']

        self.trip_encoder = HypRelEncoder("trip", data, config)
        self.qual_encoder = HypRelEncoder("qual", data, config)
        self.both_encoder = HypRelEncoder("both", data, config, entity_embs=self.qual_encoder.entity_embs, rel_embs=self.qual_encoder.rel_embs)
        self.decoder = Transformer(config)

        self.trip_encoder.to(self.device)
        self.qual_encoder.to(self.device)
        self.both_encoder.to(self.device)
        self.decoder.to(self.device)

        self.loss = torch.nn.BCELoss()


    def forward(self, sub, rel, quals):
        """
        1. Forward each encoder
        2. Concat
        3. Decode final
        """
        # Encode
        sub_trip_emb, rel_trip_emb, _, _, _ = self.trip_encoder(sub, rel, quals)
        _, _, _, _, _ = self.qual_encoder(sub, rel, quals)
        sub_both_emb, rel_both_emb, qual_obj_emb, qual_rel_emb, x = self.both_encoder(sub, rel, quals)

        # entity_embs = self.beta * sub_trip_emb + (1 - self.beta) * sub_both_emb
        # rel_embs = self.beta * rel_trip_emb + (1 - self.beta) * rel_both_emb

        scores = self.decoder(sub_trip_emb, sub_both_emb, rel_trip_emb, qual_obj_emb, qual_rel_emb, x, quals=quals)

        return scores
        


