import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_embedding_model import BaseModelForEmbedding

class CodeT5PEmbeddingModel(BaseModelForEmbedding):

    def get_hidden_state(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)
    
    def get_pooling(self, hidden_state, _):
        return hidden_state
    

class CodeT5PModel(BaseModelForEmbedding):

    def get_hidden_state(self, input_ids, attention_mask):
        outputs = self.embedding(input_ids=input_ids, attention_mask=attention_mask,
                               decoder_input_ids=input_ids, decoder_attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        return hidden_states
