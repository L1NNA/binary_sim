import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers.pooling import causal_pooling
from layers.loss import info_nce_xy

class QwenEmbeddingModel(nn.Module):
    
    def __init__(self, backbone):
        super(QwenEmbeddingModel, self).__init__()
        self.embedding = backbone
    
    def forward(self, input_ids, attention_mask, y_input_ids, y_attention_mask):
        outputs = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        x_embs = causal_pooling(outputs.last_hidden_state)
        if y_input_ids is not None:
            outputs = self.embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
            y_embs = causal_pooling(outputs.last_hidden_state)
            preds = None
            loss = info_nce_xy(x_embs, y_embs)
        else:
            preds = x_embs
            loss = None
        return {'preds':preds, 'loss': loss}