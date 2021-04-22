import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_gcn import get_param
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ConvE(nn.Module):
    def __init__(self, params):
        super(ConvE, self).__init__()
        self.p = params

        self.bn0  = torch.nn.BatchNorm2d(1)
        self.bn1  = torch.nn.BatchNorm2d(self.p['num_filt'])
        self.bn2  = torch.nn.BatchNorm1d(self.p['embed_dim'])
        
        self.hidden_drop   = torch.nn.Dropout(self.p['hid_drop'])
        self.hidden_drop2  = torch.nn.Dropout(self.p['hid_drop2'])
        self.feature_drop  = torch.nn.Dropout(self.p['feat_drop'])
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p['num_filt'], 
                                          kernel_size=(self.p['ker_sz'], self.p['ker_sz']), 
                                          stride=1, padding=0, bias=self.p['bias'])

        flat_sz_h  = int(2*self.p['k_w']) - self.p['ker_sz'] + 1
        flat_sz_w  = self.p['k_h'] - self.p['ker_sz'] + 1
        self.flat_sz = flat_sz_h*flat_sz_w*self.p['num_filt']
        self.fc  = torch.nn.Linear(self.flat_sz, self.p['embed_dim'])


    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        e1_embed = e1_embed.view(-1, 1, self.p['embed_dim'])
        rel_embed = rel_embed.view(-1, 1, self.p['embed_dim'])

        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2*qual_rel_embed.shape[1], qual_rel_embed.shape[2])

        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1)  # [bs, 2 + num_qual_pairs, emb_dim]

        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p['k_w'], self.p['k_h']))
        
        return stack_inp


    def forward(self, sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out):
        """
        Take input of encoder!!!
        """
        # Decoder
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, encoder_out.transpose(1, 0))
        score = torch.sigmoid(x)

        return score



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

        encoder_layers = TransformerEncoderLayer(self.emb_dim, self.num_heads, self.num_hidden, self.hid_drop)
        self.encoder = TransformerEncoder(encoder_layers, params['MODEL']['T_LAYERS'])


        # TODO: Fix!!!!
        self.position_embeddings = nn.Embedding(params['MAX_QPAIRS'] - 1, self.emb_dim)

        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, ent_emb_trip, ent_emb_qual, rel_embed, qual_rel_embed, qual_obj_embed):
        """
        Concat all to be passed to Transformer

        See below explanantions
        """
        ent_emb_trip = ent_emb_trip.view(-1, 1, self.emb_dim)
        ent_emb_qual = ent_emb_qual.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        
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
        stack_inp = torch.cat([ent_emb_trip, ent_emb_qual, rel_embed, quals], 1).transpose(1, 0)

        return stack_inp


    def forward(self, ent_emb_trip, ent_emb_qual, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out, quals=None):
        """
        1. Rearrange entity, rel, and quals
        2. Add positional
        3. Pass through transformer 
        """
        stk_inp = self.concat(ent_emb_trip, ent_emb_qual, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        # mask which shows which entities were padded - for future purposes, True means to mask (in transformer)
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py : 3770
        # so we first initialize with False


        # TODO: Fix!!!
        #mask = torch.zeros((ent_emb_trip.shape[0], quals.shape[1] + 3)).bool().to(self.device)  # 3 for 2 entities and 1 relation
        mask = torch.zeros((ent_emb_trip.shape[0], quals.shape[1] + 2)).bool().to(self.device)  # 3 for 2 entities and 1 relation
        
        # Put True where qual entities and relations are actually padding index 0.
        mask[:, 3:] = quals == 0
        
        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        
        # self.pooling == "avg":
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        
        x = torch.mm(x, encoder_out.transpose(1, 0))
        score = torch.sigmoid(x)

        return score