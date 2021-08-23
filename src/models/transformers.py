from numpy.lib.arraypad import pad
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.torch_transformer import TransformerEncoderBias, TransformerEncoderLayerBias

from utils.utils_gcn import *


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

        self.src_mask = params['MODEL']['SRC_MASK']

        self.batch_size = params['BATCH_SIZE']
        self.hid_drop = params['MODEL']['TRANSFORMER_DROP']
        self.num_heads = params['MODEL']['T_N_HEADS']
        self.num_hidden = params['MODEL']['T_HIDDEN']
        self.emb_dim = params['EMBEDDING_DIM']
        self.pooling = params['MODEL']['POOLING']  # avg / concat

        if params['EDGE_BIAS']:
            encoder_layers = TransformerEncoderLayerBias(self.emb_dim, self.num_heads, self.num_hidden, self.hid_drop)
            self.encoder = TransformerEncoderBias(encoder_layers, params['MODEL']['T_LAYERS'], self.num_heads, self.emb_dim, device=self.device, seq_len=params['MAX_QPAIRS'])
        else:
            encoder_layers = TransformerEncoderLayer(self.emb_dim, self.num_heads, self.num_hidden, self.hid_drop)
            self.encoder = TransformerEncoder(encoder_layers, params['MODEL']['T_LAYERS'])

        self.pos = nn.Embedding(num_positions, self.emb_dim)


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
        pos_embeddings = self.pos(positions).transpose(1, 0)
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
        self.classifier = torch.nn.Linear(self.emb_dim, self.emb_dim)



    def concat(self, head_embs, rel_embed, qual_rel_embed, qual_obj_embed, mask_embs, mask_pos, tail_embs=None):
        """
        Concat all to be passed to Transformer
        """
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        head_embs = head_embs.view(-1, 1, self.emb_dim)
        mask_embs = mask_embs.view(-1, 1, self.emb_dim)
        tail_embs = tail_embs.view(-1, 1, self.emb_dim) if tail_embs is not None else None

        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2)
        quals = quals.view(-1, 2 * qual_rel_embed.shape[1], qual_rel_embed.shape[2])

        if tail_embs is None:
            stk_inp = torch.cat([head_embs, rel_embed, mask_embs, quals], 1).transpose(1, 0)
        else:
            stk_inp = torch.cat([head_embs, rel_embed, tail_embs, quals], 1)
            stk_inp[range(stk_inp.shape[0]), mask_pos] = mask_embs.reshape(mask_embs.shape[0], mask_embs.shape[2])  # reshape since of form (a, 1, c)
            stk_inp = stk_inp.transpose(1, 0)
        
        return stk_inp


    def forward(self, head_embs, rel_emb, qual_obj_emb, qual_rel_emb, encoder_out, sample_shape, quals_ix=None, tail_embs=None, aux_mask=None):
        """
        1. Rearrange entity, rel, and quals
        2. Add positional
        3. Pass through transformer 
        """
        batch_size = head_embs.shape[0]

        if aux_mask is None:
            mask_pos = torch.ones(sample_shape[0], dtype=torch.long, device=self.device) * 2   # 2 since subject
        else:
            mask_pos = aux_mask

        # Mask empty quals
        padding_mask = torch.zeros((sample_shape[0], self.seq_length)).bool().to(self.device)
        padding_mask[:, 3:] = quals_ix == 0

        # To insert in input sequence
        input_mask = self.mask_emb.repeat(sample_shape[0], 1)
     
        # Returns Shape of (Sequence Len, Batch Size, Embedding Dim)
        trans_inp = self.concat(head_embs, rel_emb, qual_rel_emb, qual_obj_emb, input_mask, mask_pos, tail_embs)

        # Add positional
        positions = torch.arange(trans_inp.shape[0], dtype=torch.long, device=self.device).repeat(trans_inp.shape[1], 1)
        pos_embeddings = self.pos(positions).transpose(1, 0)
        trans_inp = trans_inp + pos_embeddings

        # Initial mask. Where the position we are masking can't attend
        mask = torch.zeros((batch_size, self.seq_length, self.seq_length), dtype=torch.bool, device=self.device)
        for sample in range(mask.shape[0]):
            mask[sample, :, mask_pos[sample]] = True
            mask[sample, mask_pos[sample], mask_pos[sample]] = False

        # Concat mask across all transformer heads
        head_masks = mask
        for _ in range(self.num_heads - 1):
            head_masks = torch.cat((head_masks, mask))

        x = self.encoder(trans_inp, mask=head_masks, src_key_padding_mask=padding_mask)

        # From (SeqLength, BS, Dim) to (BS, SeqLength, Dim)
        x = x.transpose(1, 0)

        # Take correct mask position for each sample
        x = x[range(x.shape[0]), mask_pos] 

        x = self.classifier(x)
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
        pos_embeddings = self.pos(positions).transpose(1, 0)
        stk_inp = stk_inp + pos_embeddings
        
        x = self.encoder(stk_inp)
        
        # x = torch.mean(x, dim=0)
        x = x.transpose(1, 0).reshape(-1, self.flat_sz)

        x = self.fc(x)

        return x
