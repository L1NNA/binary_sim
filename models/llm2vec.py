from typing import List, Optional, Tuple, Union

from transformers import Qwen2Model, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    Qwen2MLP,
)
from transformers.utils import logging

import torch
from torch import nn


logger = logging.get_logger(__name__)


class ModifiedQwen2Attention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2FlashAttention2(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2SdpaAttention(Qwen2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


QWEN2_ATTENTION_CLASSES = {
    "eager": ModifiedQwen2Attention,
    "flash_attention_2": ModifiedQwen2FlashAttention2,
    "sdpa": ModifiedQwen2SdpaAttention,
}


class ModifiedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen2BiModel(Qwen2Model):
    _no_split_modules = ["ModifiedQwen2DecoderLayer"]

    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedQwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2BiForMNTP(Qwen2ForCausalLM):
    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

from models.base_embedding_model import EmbeddingMixin

class Qwen2MNTPForSequenceEmbedding(Qwen2BiModel, EmbeddingMixin):

    def get_hidden_state(self, input_ids, attention_mask):
        return super(Qwen2BiModel, self).forward(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return super(EmbeddingMixin, self).embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )
    
    @classmethod
    def from_causal_lm(cls, causalLM:Qwen2BiForMNTP):
        config = causalLM.config
        model = cls(config)
        # be careful about the device:cuda,cpu or data parallel
        model.load_state_dict(causalLM.model.state_dict())
        return model
    

class Qwen2MNTPCausalForSequenceEmbedding(Qwen2BiForMNTP, EmbeddingMixin):

    def __init__(self, config):
        super().__init__(config)
        self.lm_head.requires_grad_ = False

    def get_hidden_state(self, input_ids, attention_mask):
        hidden_states =  super(Qwen2MNTPCausalForSequenceEmbedding, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True 
        )
        return hidden_states[-1]

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return super(EmbeddingMixin, self).embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )