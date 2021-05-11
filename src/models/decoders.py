import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_gcn import get_param
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.p = params
        self.device = params['DEVICE']

        self.batch_size = params['BATCH_SIZE']
        self.hid_drop = params['MODEL']['TRANSFORMER_DROP']
        self.num_heads = params['MODEL']['T_N_HEADS']
        self.num_hidden = params['MODEL']['T_HIDDEN']
        self.emb_dim = params['EMBEDDING_DIM']
        self.positional = params['MODEL']['POSITIONAL']
        self.pooling = params['MODEL']['POOLING']  # min / avg / concat
        self.emb_type = params['MODEL']['EMB_TYPE'].lower()

        encoder_layers = TransformerEncoderLayer(self.emb_dim, self.num_heads, self.num_hidden, self.hid_drop)
        self.encoder = TransformerEncoder(encoder_layers, params['MODEL']['T_LAYERS'])

        if self.emb_type == "same":
            self.position_embeddings = nn.Embedding(params['MAX_QPAIRS'] - 1, self.emb_dim)
        else:
            self.position_embeddings = nn.Embedding(params['MAX_QPAIRS'], self.emb_dim)

        # self.position_embeddings = nn.Embedding(params['MAX_QPAIRS'] + 1, self.emb_dim)


        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, ent_embs, rel_embed, qual_rel_embed, qual_obj_embed):
        """
        Concat all to be passed to Transformer

        See below explanantions
        """
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)

        # Happens when we want to concat multiple entity embeddings
        if isinstance(ent_embs, list):
            ent_embs = [ent_emb.view(-1, 1, self.emb_dim) for ent_emb in ent_embs]
        else:
            ent_embs = ent_embs.view(-1, 1, self.emb_dim)

        
        """
        Concat pairs on same row
        qv1, qr1
        qv2, qr2
        .   .
        .   .
        qvn, qrn
        
        Shape -> (batch_size, qual_pairs, dim*2)
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2)

        """
        Reshape such that
        qv1,
        qr1
        qv2, 
        qr2
        .
        . 
        qvn, 
        qrn
        
        Shape -> (batch_size, qual_pairs * 2, dim)
        """
        quals = quals.view(-1, 2 * qual_rel_embed.shape[1], qual_rel_embed.shape[2])

        """
        [Batch of trip entities]
        [Batch of qual entities]
        [Batch of relations]
        [Batch of qv1's]
        [Batch of qr2's]
        .
        .

        Shape -> (qual_pairs*2 + ent_embeds + rel_embed, batch_size, dim)
        """

        if isinstance(ent_embs, list):
            return torch.cat([*ent_embs, rel_embed, quals], 1).transpose(1, 0)
        else:
            return torch.cat([ent_embs, rel_embed, quals], 1).transpose(1, 0)



    def forward(self, ent_embs, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out, ents=None, quals=None):
        """
        1. Rearrange entity, rel, and quals
        2. Add positional
        3. Pass through transformer 
        """
        stk_inp = self.concat(ent_embs, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        # Mask qualifier pairs that are empty
        # e.g. If 2 pairs then mask[:, 2+4:] = True, otherwise False
        mask_pos = 2  # = Num entities + 1 relation
        mask = torch.zeros((ents.shape[0], quals.shape[1] + mask_pos)).bool().to(self.device)
        mask[:, mask_pos:] = quals == 0
        
        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        
        x = torch.mm(x, encoder_out.transpose(1, 0))
        score = torch.sigmoid(x)

        return score