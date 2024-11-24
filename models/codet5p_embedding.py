import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.pooling import causal_pooling
from layers.loss import info_nce_xy

class CodeT5PEmbeddingModel(nn.Module):
    
    def __init__(self, backbone):
        super(CodeT5PEmbeddingModel, self).__init__()
        self.embedding = backbone
        self.dim = self.embedding.config.embed_dim
        self.dtype = self.embedding.dtype
    
    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        # input_ids: b x L x d -> hidden_states: b x d
        x_emb = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        if y_input_ids is not None:
            y_emb = self.embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
            loss = info_nce_xy(x_emb, y_emb, labels)
        else:
            loss = 0
        return {'preds':x_emb, 'loss': loss}
    
