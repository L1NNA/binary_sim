import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import PositionalEncoding
from layers.pooling import mask_mean_pooling, causal_pooling
from layers.loss import info_nce
from layers.attention import TransformerLayer, CausalTransformerLayer
from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Union, Tuple
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from einops import einsum, rearrange

## TODO
## attention mask(check), ALIBI positional encoding(check), causal loss(check), KV cache(check), activation functions for FFN, online safe softmax?
## implement LM head and generation logic(check, using generationmixin), retrain the model on paired causal data

class CacheCausal(Cache):
    def __init__(self):
        super().__init__()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.get(layer_idx).shape[1]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

class CustomConfig(PretrainedConfig):
    model_type = "custom_model"

    def __init__(
        self,
        vocab_size=50_000,
        hidden_size=1024,
        num_hidden_layers=12,
        kv_heads = 2,
        num_attention_heads=8,
        intermediate_size=4096,
        dropout = 0.1,
        device = 'cuda',
        dtype = 'torch.bfloat16',
        causal_lm = True,
        max_len = 512,
        use_cache=False,
        layer_norm_eps = 1e-5,
        initializer_range=0.02,
        tie_word_embeddings = False,
        use_flash_attn = False,
        gamma_init = 1.0,
        return_dict = True,
        use_alibi = True,
        use_blk_mask = True,
        **kwargs
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.kv_heads = kv_heads
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.use_cache = use_cache
        self.causal_lm = causal_lm
        self.max_len = max_len
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_flash_attn = use_flash_attn
        self.gamma_init = gamma_init
        self.return_dict = return_dict
        self.use_alibi = use_alibi
        self.use_blk_mask = use_blk_mask
        super().__init__(tie_word_embeddings = tie_word_embeddings, **kwargs)

class CustomPretrainedModel(PreTrainedModel):
    config_class = CustomConfig
    base_model_prefix = 'model'
    _no_split_modules = ["CausalTransformerLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class CustomModel(CustomPretrainedModel):
    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.attention_layers = nn.ModuleList([CausalTransformerLayer(config,
                                                                      layer_idx,
                                                                      ) for layer_idx in range(config.num_hidden_layers)])
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size, 
                                    #   device = config.device, dtype = torch.bfloat16
                                      )
        self.max_len = config.max_len
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, 
            # device=config.device, dtype=torch.bfloat16
        )
        if not config.use_alibi:
            self.pos_enc = PositionalEncoding(config.hidden_size)

        self.post_init()
    
    def forward(self, input_ids = None, 
                attention_mask = None, 
                position_ids = None, 
                past_key_values = None, 
                inputs_embeds = None, 
                use_cache = None, 
                output_attentions = None, 
                output_hidden_states = None, 
                return_dict = None,
                use_flash_attn = None,
                use_alibi = None,
                cache_position = None,
                blk_mask = None,
                ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_flash_attn = use_flash_attn if use_flash_attn is not None else self.config.use_flash_attn
        use_alibi = use_alibi if use_alibi is not None else self.config.use_alibi


        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            b_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            b_size, seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        
        assert seq_len <= self.max_len, f"Input sequence length {seq_len} exceeds the maximum length {self.max_len}"

        past_key_values_length = 0

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
        #     )
        #     position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        # else:
        #     position_ids = position_ids.view(-1, seq_len).long()

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        causal_mask = rearrange(causal_mask, 'b i q k -> b () i q k')

        hidden_states = inputs_embeds
        if not use_alibi:
            hidden_states += self.pos_enc(hidden_states)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for i, attn_layer in enumerate(self.attention_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = attn_layer(hidden_states, 
                                       attention_mask = causal_mask, 
                                       position_ids = position_ids, 
                                       past_key_values = past_key_values, 
                                       output_attentions = output_attentions,
                                       use_cache = use_cache, 
                                       use_flash_attn = use_flash_attn,
                                       use_alibi = use_alibi,
                                       blk_mask = blk_mask,
                                       )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if return_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and not (using_static_cache or using_sliding_window_cache)
        #     and not output_attentions
        # ):
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         sliding_window=self.config.sliding_window,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: CustomConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            # if config.sliding_window is not None:
            #     # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
            #     # the check is needed to verify is current checkpoint was trained with sliding window or not
            #     if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
            #         sliding_attend_mask = torch.arange(target_length, device=device) <= (
            #             cache_position.reshape(-1, 1) - config.sliding_window
            #         )
            #         diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class CustomModelCausal(CustomPretrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        # Define your custom model architecture here
        # if config.dtype == 'bfloat16':
        #     self.dtype = torch.bfloat16
        # elif config.dtype == 'float32':
        #     self.dtype = torch.float32
        self.model = CustomModel(config)
        self.vocab_size = config.vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                use_flash_attn = None,
                use_alibi = None,
                cache_position = None,
                blk_mask = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_flash_attn = use_flash_attn if use_flash_attn is not None else self.config.use_flash_attn
        use_alibi = use_alibi if use_alibi is not None else self.config.use_alibi

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_flash_attn=use_flash_attn,
            cache_position = cache_position,
            blk_mask = blk_mask,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    



    
