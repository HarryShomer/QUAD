from .encoders import *
from .transformers import *
from utils.utils_gcn import *


class HypRelModel(nn.Module):

    def __init__(self, data, config):
        super(self.__class__, self).__init__()

        self.config  = config
        self.device  = config['DEVICE']
        self.opn     = config['MODEL']['OPN']
        self.num_ent = config['NUM_ENTITIES']
        self.num_rel = config['NUM_RELATIONS']
        self.emb_dim = config['EMBEDDING_DIM']
        self.beta    = config['BETA']

        self.ent_embs = get_param((self.num_ent, self.emb_dim))
        self.rel_embs = self.get_rel_emb()

        self.trip_encoder = HypRelEncoder(data, config, config['MODEL']['TRIP_LAYERS'])
        self.trip_encoder.to(self.device)

        self.qual_encoder = HypRelEncoder(data, config, config['MODEL']['QUAL_LAYERS'], qual=True)
        self.qual_encoder.to(self.device)

        self.decoder = TransformerDecoder(config)
        self.decoder.to(self.device)

        self.loss = torch.nn.BCELoss()


    def get_rel_emb(self):
        """
        Differs for RotatE. Otherwise just use `get_param`
        """
        if self.opn.lower() == "rotate":
            return get_rotate_param(self.num_rel, self.emb_dim)
        else:
            return get_param((self.num_rel * 2, self.emb_dim))


    def index_embs(self, x, r, ent_ix, rel_ix, quals_ix):
        """
        """
        sub_emb = torch.index_select(x, 0, ent_ix)
        rel_emb = torch.index_select(r, 0, rel_ix)

        # flatten quals
        quals_ents = quals_ix[:, 1::2].view(1,-1).squeeze(0)
        quals_rels = quals_ix[:, 0::2].view(1,-1).squeeze(0)

        qual_obj_emb = torch.index_select(x, 0, quals_ents)
        qual_rel_emb = torch.index_select(r, 0, quals_rels)

        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb


    def forward(self, ent_ix, rel_ix, quals_ix):
        """
        1. Get embeddings (if bases or not)
        2. Forward each encoder
        3. Decode final
        """
        init_ent = self.ent_embs
        init_rel = self.rel_embs

        # x1, r1 = self.trip_encoder("trip", ent_ix, rel_ix, quals_ix, init_ent, init_rel)

        # if not self.config['ONLY-TRIPS']:
        #     x2, r2 = self.qual_encoder("qual", ent_ix, rel_ix, quals_ix, x1, r1)
        #     x = self.beta * x1 + (1 - self.beta) * x2
        #     r = self.beta * r1 + (1 - self.beta) * r2
        # else:
        #     x, r = x1, r1

        x, r = self.trip_encoder("trip", ent_ix, rel_ix, quals_ix, init_ent, init_rel)

        if not self.config['ONLY-TRIPS']:
            x, r = self.qual_encoder("qual", ent_ix, rel_ix, quals_ix, x, r)
    
        e_emb, r_emb, qe_emb, qr_emb = self.index_embs(x, r, ent_ix, rel_ix, quals_ix)

        return self.decoder(e_emb, r_emb, qe_emb, qr_emb, x, ents=ent_ix, quals=quals_ix)
