from .new_encoders import *
from .transformers import *
from utils.utils_gcn import *



class HypRelModel(nn.Module):

    def __init__(self, data, config):
        super(self.__class__, self).__init__()

        self.config    = config
        self.device    = config['DEVICE']
        self.mask      = config['MODEL']['SRC_MASK']
        self.opn       = config['MODEL']['OPN']
        self.num_ent   = config['NUM_ENTITIES']
        self.num_rel   = config['NUM_RELATIONS']
        self.emb_dim   = config['EMBEDDING_DIM']

        #self.ent_skip_matrix = get_param((self.emb_dim * 2, self.emb_dim))

        self.ent_embs = get_param((self.num_ent, self.emb_dim))
        self.rel_embs = self.get_rel_emb()

        # self.trip_encoder = HypRelEncoder(data, config, config['MODEL']['TRIP_LAYERS'])
        # self.trip_encoder.to(self.device)

        # self.qual_encoder = HypRelEncoder(data, config, config['MODEL']['QUAL_LAYERS'], qual=True)
        # self.qual_encoder.to(self.device)

        self.hyp_encoder = HypRelEncoder(data, config, config['MODEL']['TRIP_LAYERS'], qual=True)
        self.hyp_encoder.to(self.device)


        self.decoder = MaskedTransformerDecoder(config) if self.mask else TransformerDecoder(config)
        self.decoder.to(self.device)

        self.loss_fn = torch.nn.BCELoss()


    def loss(self, preds, labels):
        return self.loss_fn(preds, labels)


    def get_rel_emb(self):
        """
        Differs for RotatE. Otherwise just use `get_param`
        """
        if self.opn.lower() == "rotate":
            return get_rotate_param(self.num_rel, self.emb_dim)
        else:
            return get_param((self.num_rel * 2, self.emb_dim))


    def index_embs(self, x, r, sub_ix, rel_ix, quals_ix):
        """
        Index entity and relation embeddings from matrices
        """
        sub_emb = torch.index_select(x, 0, sub_ix)
        rel_emb = torch.index_select(r, 0, rel_ix)

        # Flatten quals
        quals_ents = quals_ix[:, 1::2].reshape(1, -1).squeeze(0)
        quals_rels = quals_ix[:, 0::2].reshape(1, -1).squeeze(0)

        qual_obj_emb = torch.index_select(x, 0, quals_ents)
        qual_rel_emb = torch.index_select(r, 0, quals_rels)

        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb


    def index_embs_aux(self, x, r, sub_ix, rel_ix, obj_ix, quals_ix):
        """
        Include obj index
        """
        obj_emb = torch.index_select(x, 0, obj_ix)

        return *self.index_embs(x, r, sub_ix, rel_ix, quals_ix), obj_emb


    def forward(self, sub_ix, rel_ix, quals_ix, aux_ent=None, aux_rel=None):
        """
        1. Get embeddings (if bases or not)
        2. Forward each encoder
        3. Decode final
        """
        init_ent = self.ent_embs
        init_rel = self.rel_embs
        aux_ent_preds, aux_rel_preds = None, None

        x, r = self.hyp_encoder(init_ent, init_rel)

        # Subject Prediction
        s_emb, r_emb, qe_emb, qr_emb = self.index_embs(x, r, sub_ix, rel_ix, quals_ix)
        obj_preds = self.decoder(s_emb, r_emb, qe_emb, qr_emb, x, sub_ix.shape, quals_ix=quals_ix)

        # Qual Entity prediction if included
        if aux_ent is not None:            
            s_emb, r_emb, qe_emb, qr_emb, o_emb = self.index_embs_aux(x, r, aux_ent['base_sub_ix'], aux_ent['base_rel_ix'], aux_ent['base_obj_ix'], aux_ent['quals'])
            aux_ent_preds = self.decoder(s_emb, r_emb, qe_emb, qr_emb, x, aux_ent['base_sub_ix'].shape, quals_ix=aux_ent['quals'], tail_embs=o_emb, aux_mask=aux_ent['mask'])

        # Aux relation pred if included
        if aux_rel is not None:
            s_emb, r_emb, qe_emb, qr_emb, o_emb = self.index_embs_aux(x, r, aux_rel['base_sub_ix'], aux_rel['base_rel_ix'], aux_rel['base_obj_ix'], aux_rel['quals'])
            aux_rel_preds = self.decoder(s_emb, r_emb, qe_emb, qr_emb, r, aux_rel['base_sub_ix'].shape, quals_ix=aux_rel['quals'], tail_embs=o_emb, aux_mask=aux_rel['mask'])
        

        return obj_preds, aux_ent_preds, aux_rel_preds
        

