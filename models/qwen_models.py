import torch
import torch.nn as nn
from transformers import Qwen2Model, Qwen2ForCausalLM

from models.base_embedding_model import EmbeddingMixin, EmbeddingOutput
from layers.pooling import attention_mask_pooling, mask_mean_pooling
from layers.loss import info_nce, gte_info_nce
from models.llm2vec_custom import CustomQwen2BiModel


class Qwen2ForSequenceEmbedding(Qwen2Model, EmbeddingMixin):

    def get_hidden_state(self, input_ids, attention_mask, blk_mask=None):
        return super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            blk_mask=blk_mask,
        ).last_hidden_state

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None, **kwargs):
        return self.embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels, **kwargs
        )
    
    def get_pooling(self, hidden_state, attention_mask, blk_mask=None, pooling='attention_mask_pooling'):
        if pooling == 'attention_mask_pooling':
            return attention_mask_pooling(hidden_state, attention_mask)
        elif pooling == 'mask_mean_pooling':
            if blk_mask is not None:
                mask = attention_mask * blk_mask
                return mask_mean_pooling(hidden_state, mask)
    
    @classmethod
    def from_causal_lm(cls, causalLM:Qwen2ForCausalLM):
        config = causalLM.config
        model = cls(config)
        # be careful about the device:cuda,cpu or data parallel
        model.load_state_dict(causalLM.model.state_dict())
        return model
    
    def get_loss(self, x_emb, y_emb, labels):
        if y_emb is None:
            return None
        return gte_info_nce(x_emb, y_emb, labels)

class CustomQwen2ForSequenceEmbedding(CustomQwen2BiModel, EmbeddingMixin):
    def __init__(self, config):
        super().__init__(config)
        self.use_unsupervised = getattr(config, 'use_unsupervised', False)

    def get_hidden_state(self, input_ids, attention_mask, blk_mask=None, arch=None):
        return super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            blk_mask=blk_mask,
            dropout = 0.1,
            arch=arch,
        ).last_hidden_state

    def forward(self, input_ids, 
                attention_mask, 
                blk_mask=None,  
                arch=None,
                y_input_ids=None, 
                y_attention_mask=None, 
                y_blk_mask=None, 
                y_arch=None,
                labels=None):
        if self.use_unsupervised and self.training:
            return self.embeddingWithUnsupervised(
                input_ids, 
                attention_mask, 
                blk_mask=blk_mask,
                arch=arch,
                y_input_ids=y_input_ids, 
                y_attention_mask=y_attention_mask,
                y_blk_mask=y_blk_mask, 
                y_arch=y_arch,
                labels=labels,
            )
        else:
            return self.embedding(
                input_ids, 
                attention_mask, 
                blk_mask=blk_mask,
                arch=arch,
                y_input_ids=y_input_ids, 
                y_attention_mask=y_attention_mask,
                y_blk_mask=y_blk_mask, 
                y_arch=y_arch,
                labels=labels,
            )
    
    def get_pooling(self, hidden_state, attention_mask, blk_mask=None):
        return attention_mask_pooling(hidden_state, attention_mask)
        # if blk_mask is not None:
        #     mask = attention_mask * blk_mask
        # return mask_mean_pooling(hidden_state, mask)
    
    def get_loss(self, x_emb, y_emb, labels, anchor_emb=None):
        if y_emb is None:
            return None
        return gte_info_nce(x_emb, y_emb, labels, anchor_emb=anchor_emb)
    
    @classmethod
    def from_causal_lm(cls, causalLM:Qwen2ForCausalLM):
        config = causalLM.config
        model = cls(config)
        # be careful about the device:cuda,cpu or data parallel
        model.load_state_dict(causalLM.model.state_dict(), strict=False)
        return model
    
def preload_qwen2_from_causal_lm(model_path, cls, local_path, config):
    causalLM = Qwen2ForCausalLM.from_pretrained(
        model_path if local_path is None else local_path,
        torch_dtype = torch.bfloat16
    )
    model = cls(config)
    # be careful about the device:cuda,cpu or data parallel
    model.load_state_dict(causalLM.model.state_dict(), strict=False)
    return model

# class Qwen2CausalForSequenceEmbedding(Qwen2ForCausalLM, EmbeddingMixin):

#     def __init__(self, config):
#         super().__init__(config)
#         self.lm_head.requires_grad_ = False

#     def get_hidden_state(self, input_ids, attention_mask):
#         hidden_states =  super(Qwen2ForCausalLM, self).forward(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True 
#         )
#         return hidden_states[-1]

#     def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
#         return self.embedding(
#             input_ids, attention_mask, y_input_ids, y_attention_mask, labels
#         )
