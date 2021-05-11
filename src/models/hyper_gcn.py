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

        self.ent_embs = get_param((self.num_ent, self.emb_dim))
        self.rel_embs = get_param((self.num_rel * 2, self.emb_dim))

        self.encoder = HypRelEncoder(data, config)
        self.encoder.to(self.device)

        self.decoder = Transformer(config)
        self.decoder.to(self.device)

        self.loss = torch.nn.BCELoss()



    def forward(self, ent_ix, rel_ix, quals_ix):
        """
        Encode & Decode
        """
        init_ent = self.ent_embs  
        init_rel = self.rel_embs

        e_emb, r_emb, qe_emb, qr_emb, x, r = self.encoder(ent_ix, rel_ix, quals_ix, init_ent, init_rel)

        return self.decoder(e_emb, r_emb, qe_emb, qr_emb, x, ents=ent_ix, quals=quals_ix)
