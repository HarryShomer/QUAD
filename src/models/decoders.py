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


        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, ent_emb_trip, ent_emb_both, rel_embed, qual_rel_embed, qual_obj_embed):
        """
        Concat all to be passed to Transformer

        See below explanantions
        """
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        ent_emb_trip = ent_emb_trip.view(-1, 1, self.emb_dim)
        ent_emb_both = ent_emb_both.view(-1, 1, self.emb_dim)
        
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
        if self.emb_type == "same":
            stack_inp = torch.cat([ent_emb_both, rel_embed, quals], 1).transpose(1, 0)
        else:
            stack_inp = torch.cat([ent_emb_trip, ent_emb_both, rel_embed, quals], 1).transpose(1, 0)


        return stack_inp


    def forward(self, ent_emb_trip, ent_emb_both, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out, quals=None):
        """
        1. Rearrange entity, rel, and quals
        2. Add positional
        3. Pass through transformer 
        """
        stk_inp = self.concat(ent_emb_trip, ent_emb_both, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        # mask which shows which entities were padded - for future purposes, True means to mask (in transformer)
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py : 3770
        # so we first initialize with False
        
        # = # entities + 1 relation
        mask_pos = 2 if self.emb_type == "same" else 3

        mask = torch.zeros((ent_emb_both.shape[0], quals.shape[1] + mask_pos)).bool().to(self.device)
        
        # Put True where qual entities and relations are actually padding index 0.
        mask[:, mask_pos:] = quals == 0
        
        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        
        # self.pooling == "avg":
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        
        x = torch.mm(x, encoder_out.transpose(1, 0))
        score = torch.sigmoid(x)

        return score