import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import PositionalEncoding
from layers.pooling import mask_mean_pooling, causal_pooling
from layers.loss import info_nce
from layers.attention import TransformerLayer

class CoECosSim(nn.Module):

    def __init__(self, backbone, n_heads=8, n_layers=4):
        super(CoECosSim, self).__init__()
        self.embedding = backbone
        dtype = self.embedding.dtype
        dim = self.embedding.config.embed_dim
        self.dtype = dtype
        self.dim = dim

        self.pe = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True, dtype=dtype)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def single_forward(self, input_ids, attention_mask):
        b, w, _ = input_ids.size()
        # b x num_blocks x num_tokens -> (b*num_blocks) x num_tokens
        input_ids = input_ids.reshape(b*w, -1)
        attention_mask = attention_mask.reshape(b*w, -1)
        embs = self.embedding(input_ids, attention_mask) # (b*num_blocks) x dim
        w_embs = embs.reshape(b, w, self.dim).contiguous() # b x num_blocks x dim
        w_embs += self.pe(w_embs)
        w_mask = attention_mask.max(-1).values.reshape(b, w).contiguous() # b x num_blocks
        w_mask = ~w_mask.bool()
        w_embs = self.decoder(w_embs, src_key_padding_mask=w_mask) # b x num_blocks x dim
        o_embs = mask_mean_pooling(w_embs, w_mask).to(self.dtype) # b x dim
        return o_embs

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        embs = self.single_forward(input_ids, attention_mask)
        loss = None
        if y_input_ids is not None:
            y_embs = self.single_forward(y_input_ids, y_attention_mask)
            pred = F.cosine_similarity(embs, y_embs)
            if labels is not None:
                loss = F.cosine_embedding_loss(embs, y_embs, labels)
        else:
            pred = embs
        return {
            "preds": pred,
            "loss": loss
        }