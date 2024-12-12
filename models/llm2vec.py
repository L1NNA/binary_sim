from typing import List, Optional, Tuple, Union

from transformers import Qwen2Model, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    Qwen2MLP,
    Qwen2RotaryEmbedding
)
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

import torch
from torch import nn
from layers.loss import gte_info_nce

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
        super().__init__(config)
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
        # self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )  # in original implementation - torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        # Commenting out next 2 lines to disable causal masking
        # if sequence_length != 1:
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length
                ].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


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

    def get_hidden_state(self, input_ids, attention_mask, **kwargs):
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None, **kwargs):
        return self.embedding(
            input_ids, attention_mask, y_input_ids=y_input_ids, y_attention_mask=y_attention_mask, labels=labels
        )
    
    def get_loss(self, x_emb, y_emb, labels):
        if y_emb is None:
            return None
        return gte_info_nce(x_emb, y_emb, labels)
    
#     @classmethod
#     def from_causal_lm(cls, causalLM:Qwen2BiForMNTP):
#         config = causalLM.config
#         model = cls(config)
#         # be careful about the device:cuda,cpu or data parallel
#         model.load_state_dict(causalLM.model.state_dict())
#         return model
    

# class Qwen2MNTPCausalForSequenceEmbedding(Qwen2BiForMNTP, EmbeddingMixin):

#     def __init__(self, config):
#         super().__init__(config)
#         self.lm_head.requires_grad_ = False

#     def get_hidden_state(self, input_ids, attention_mask):
#         hidden_states =  super(Qwen2MNTPCausalForSequenceEmbedding, self).forward(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True 
#         )
#         return hidden_states[-1]

#     def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
#         return super(EmbeddingMixin, self).embedding(
#             input_ids, attention_mask, y_input_ids, y_attention_mask, labels
#         )