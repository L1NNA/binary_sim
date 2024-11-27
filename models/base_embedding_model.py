import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, Qwen2Config
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from dataclasses import dataclass

from layers.pooling import cls_pooling, mean_pooling, mask_mean_pooling, attention_mask_pooling
from layers.loss import info_nce


@dataclass
class EmbeddingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embedding: torch.FloatTensor = None
    y_embedding: torch.FloatTensor = None


class EmbeddingMixin:

    def get_hidden_state(self, input_ids, attention_mask):
        raise 'Not implemented'
    
    def get_pooling(self, hidden_state, attention_mask):
        pooling = self.config.__getattribute__('pooling')
        if pooling is None:
            pooling = 'mean'
        
        if pooling == 'mean':
            return mean_pooling(hidden_state)
        elif pooling == 'cls':
            return cls_pooling(hidden_state)
        elif pooling == 'causal':
            return attention_mask_pooling(hidden_state, attention_mask)
        elif pooling == 'mask_mean':
            return mask_mean_pooling(hidden_state, attention_mask)
        else:
            raise f'Unsupported pooling: {pooling}'
        
    def get_loss(self, x_emb, y_emb, labels):
        if y_emb is None:
            return None
        return info_nce(x_emb, y_emb, labels)

    def embedding(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        hidden_state = self.get_hidden_state(input_ids, attention_mask)
        x_emb = self.get_pooling(hidden_state, attention_mask)

        y_emb = None
        if y_input_ids is not None:
            hidden_state = self.get_hidden_state(y_input_ids, y_attention_mask)
            y_emb = self.get_pooling(hidden_state, y_attention_mask)

        loss = self.get_loss(x_emb, y_emb, labels)
        return EmbeddingOutput(
            loss=loss,
            embedding=x_emb,
            y_embedding=y_emb
        )



class BaseModelForEmbedding(PreTrainedModel):

    def __init__(self, config:PretrainedConfig) -> None:
        super(BaseModelForEmbedding, self).__init__(config)
        self.backbone = self.get_model()

    def get_model(self):
        """Please override"""
        return AutoModel.from_pretrained(self.config.name_or_path)
    
    def get_hidden_state(self, input_ids, attention_mask):
        """Please override"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def get_pooling(self, hidden_state, attention_mask):
        """Please override"""
        pooling = self.config.__getattribute__('pooling')
        if pooling is None:
            pooling = 'mean'
        
        if pooling == 'mean':
            return mean_pooling(hidden_state)
        elif pooling == 'cls':
            return cls_pooling(hidden_state)
        elif pooling == 'causal':
            return attention_mask_pooling(hidden_state, attention_mask)
        elif pooling == 'mask_mean':
            return mask_mean_pooling(hidden_state, attention_mask)
        else:
            raise f'Unsupported pooling: {pooling}'
        
    def get_loss(self, x_emb, y_emb, labels):
        if y_emb is None:
            return None
        return info_nce(x_emb, y_emb, labels)


    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        hidden_state = self.get_hidden_state(input_ids, attention_mask)
        x_emb = self.get_pooling(hidden_state, attention_mask)

        y_emb = None
        if y_input_ids is not None:
            hidden_state = self.get_hidden_state(y_input_ids, y_attention_mask)
            y_emb = self.get_pooling(hidden_state, y_attention_mask)

        loss = self.get_loss(x_emb, y_emb, labels)
        return EmbeddingOutput(
            loss=loss,
            embedding=x_emb,
            y_embedding=y_emb
        )