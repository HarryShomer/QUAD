from utils.utils_gcn import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    
    def __init__(self, params, num_positions):
        """
        General class for a Transformer

        Parameters:
        -----------
            params: dict
                General parameters. Mostly cmd line args
            num_positions: int
                Number of positional embeddings 
        """
        super().__init__()

        self.p = params
        self.device = params['DEVICE']

        self.batch_size = params['BATCH_SIZE']
        self.hid_drop = params['MODEL']['TRANSFORMER_DROP']
        self.num_heads = params['MODEL']['T_N_HEADS']
        self.num_hidden = params['MODEL']['T_HIDDEN']
        self.emb_dim = params['EMBEDDING_DIM']
        self.pooling = params['MODEL']['POOLING']  # min / avg / concat

        encoder_layers = TransformerEncoderLayer(self.emb_dim, self.num_heads, self.num_hidden, self.hid_drop)

        self.encoder = TransformerEncoder(encoder_layers, params['MODEL']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(num_positions, self.emb_dim)


class TransformerDecoder(Transformer):
    """
    Transformer used to perform KG Completion

    Takes entity, relation, and qual pairs
    """
    def __init__(self, params):
        super().__init__(params, params['MAX_QPAIRS'] - 1)
        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, ent_embs, rel_embed, qual_rel_embed, qual_obj_embed):
        """
        Concat all to be passed to Transformer

        See below explanantions
        """
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
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
        [Batch of qv1s]
        [Batch of qr2s]
        .
        .
        Shape -> (qual_pairs*2 + ent_embeds + rel_embed, batch_size, dim)
        """
        return torch.cat([ent_embs, rel_embed, quals], 1).transpose(1, 0)



    def forward(self, ent_embs, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out, sample_shape, quals_ix=None):
        """
        1. Rearrange entity, rel, and quals
        2. Add positional
        3. Pass through transformer 
        """
        # (Sequence Len, Batch Size, Embedding Dim)
        stk_inp = self.concat(ent_embs, rel_emb, qual_rel_emb, qual_obj_emb)

        positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
        pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
        stk_inp = stk_inp + pos_embeddings

        # Mask qualifier pairs that are empty
        # e.g. If 2 pairs then mask[:, 2+4:] = True, otherwise False
        mask_pos = 2  # = Num entities + 1 relation
        mask = torch.zeros((sample_shape[0], quals_ix.shape[1] + mask_pos)).bool().to(self.device)
        mask[:, mask_pos:] = quals_ix == 0
        
        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        
        x = torch.mm(x, encoder_out.transpose(1, 0))
        score = torch.sigmoid(x)

        return score



class MaskedTransformerDecoder(Transformer):
    """
    Transformer used to perform KG Completion

    Takes entity, relation, and qual pairs
    """
    def __init__(self, params):
        super().__init__(params, params['MAX_QPAIRS'])
        self.seq_length = params['MAX_QPAIRS']

        self.mask_emb = torch.nn.Parameter(torch.randn(1, self.emb_dim, dtype=torch.float32), True)
        
        self.flat_sz = self.emb_dim * (self.seq_length - 1)
        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, head_embs, rel_embed, qual_rel_embed, qual_obj_embed, mask_embs, mask_pos, tail_embs=None):
        """
        Concat all to be passed to Transformer
        """
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        head_embs = head_embs.view(-1, 1, self.emb_dim)
        mask_embs = mask_embs.view(-1, 1, self.emb_dim)
        tail_embs = tail_embs.view(-1, 1, self.emb_dim) if mask_pos == -1 else None

        # When mask_pos = -1, for each sequence we have 6 qr and 5 qv. So in order for this to work we
        # temporarily add along the 1st dimension for qv
        if mask_pos == -1:
            z = torch.zeros(qual_obj_embed.shape[0], 1, qual_obj_embed.shape[2]).to(self.device)
            qual_obj_embed = torch.cat([qual_obj_embed, z], 1)

        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2)
        quals = quals.view(-1, 2 * qual_rel_embed.shape[1], qual_rel_embed.shape[2])

        # Bec. of the previous addition we need to take it out. We know the sequence length should be 11
        if mask_pos == -1:
            quals = quals[:, :11, :]

        if mask_pos == 2:
            stk_inp = torch.cat([head_embs, rel_embed, mask_embs, quals], 1).transpose(1, 0)
        else:
            stk_inp = torch.cat([head_embs, rel_embed, tail_embs, quals, mask_embs], 1).transpose(1, 0)
        
        return stk_inp


    def forward(self, head_embs, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out, sample_shape, tail_embs=None, quals_ix=None):
        """
        1. Rearrange entity, rel, and quals
        2. Add positional
        3. Pass through transformer 
        """
        mask_pos = 2 if tail_embs is None else -1 

        # To insert in input
        ent_mask = self.mask_emb.repeat(sample_shape[0], 1)

        # Zeroes for mask to identify what to mask
        ins = torch.zeros(sample_shape, dtype=torch.bool, device=self.device)

        # Create mask and init as zeroes for empty quals.
        # Option 1 (subject): [H, R, QR1, QV1, ..., QRN, QVN] -> NUM_QUALS + 2
        # Option 2 (object):  [H, R, T, QR1, QV1, ..., QRN]   -> NUM_QUALS + 3
        qual_offset = 3 if mask_pos == -1 else 2
        mask = torch.zeros((sample_shape[0], quals_ix.shape[1] + qual_offset)).bool().to(self.device)
        mask[:, qual_offset:] = quals_ix == 0

        # When obj put in [h, r, <HERE>, qr1, qv2....qrn, qvn]
        # Otherwise put at end for qual we are predicting [h, r, t, qr1, qv2....qrn, <HERE>]
        if mask_pos == 2:
            mask = torch.cat((mask[:, :2], ins.unsqueeze(1), mask[:, 2:]), axis=1)
        else:
            mask = torch.cat((mask, ins.unsqueeze(1)), axis=1)
     
        # Returns Shape of (Sequence Len, Batch Size, Embedding Dim)
        stk_inp = self.concat(head_embs, rel_emb, qual_rel_emb, qual_obj_emb, ent_mask, mask_pos, tail_embs)

        positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
        pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
        stk_inp = stk_inp + pos_embeddings

        # Last when qual otherwise index 2
        x = self.encoder(stk_inp, src_key_padding_mask=mask)[mask_pos]
        # x = torch.mean(x, dim=0)
        x = self.fc(x)

        x = torch.mm(x, encoder_out.transpose(1, 0))
        scores = torch.sigmoid(x)

        return scores



class TransformerTriplets(Transformer):
    """
    Transformer to encode base triplet
    """
    def __init__(self, params):
        super().__init__(params, 3) # 3 = head, relation, tail

        self.flat_sz = self.emb_dim * 3
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)


    def forward(self, head_ents, tail_ents, rels):
        """
        """
        # rearrange data
        head_ents = head_ents.view(-1, 1, self.emb_dim)
        tail_ents = tail_ents.view(-1, 1, self.emb_dim)
        rels      = rels.view(-1, 1, self.emb_dim)
        stk_inp   = torch.cat([head_ents, rels, tail_ents], 1).transpose(1, 0)

        positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
        pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
        stk_inp = stk_inp + pos_embeddings
        
        x = self.encoder(stk_inp)
        
        # x = torch.mean(x, dim=0)
        x = x.transpose(1, 0).reshape(-1, self.flat_sz)

        x = self.fc(x)

        return x



# class TransformerQualifiers(Transformer):
#     """
#     Transformer to encode qualifier pairs
#     """
#     def __init__(self, params):
#         super().__init__(params, params['MAX_QPAIRS'] - 3) # 3 because subtract possible base triplet embeddings

    
#     def forward(self, qual_rel_embed, qual_obj_embed):
#         """
#         """
#         quals = torch.cat((qual_rel_embed, qual_obj_embed), 2)
#         quals = quals.view(-1, 2 * qual_rel_embed.shape[1], qual_rel_embed.shape[2])
#         stk_inp = quals.transpose(1, 0)

#         positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
#         pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
#         stk_inp = stk_inp + pos_embeddings
        
#         x = self.encoder(stk_inp, src_key_padding_mask=mask)
#         x = torch.mean(x, dim=0)

#         return x


