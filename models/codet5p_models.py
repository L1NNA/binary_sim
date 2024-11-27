import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5EncoderModel
from transformers.configuration_utils import PretrainedConfig

from models.base_embedding_model import EmbeddingMixin

class CodeT5pEmbeddingConfig(PretrainedConfig):
    model_type = "codet5p_embedding"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
            self,
            vocab_size=32103,
            d_model=768,
            embed_dim=256,
            d_kv=64,
            d_ff=3072,
            num_layers=12,
            num_heads=12,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=2,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

class CodeT5PEncoderForSequenceEmbedding(T5EncoderModel, EmbeddingMixin):

    config_class = CodeT5pEmbeddingConfig
    authorized_missing_keys = [
        r"encoder.embed_tokens.weight",
    ]

    def __init__(self, config: CodeT5pEmbeddingConfig):
        super().__init__(config)
        self.proj = nn.Linear(config.d_model, config.embed_dim)

    def get_pooling(self, hidden_state, attention_mask):
        return hidden_state

    def get_hidden_state(
            self,
            input_ids,
            attention_mask
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embedding = F.normalize(self.proj(encoder_outputs.last_hidden_state[:, 0, :]), dim=-1)
        return embedding

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return self.embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )
    

class CodeT5PForSequenceEmbedding(T5ForConditionalGeneration, EmbeddingMixin):

    def get_hidden_state(self, input_ids, attention_mask):
        outputs = super(T5ForConditionalGeneration, self).forward(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=input_ids, decoder_attention_mask=attention_mask,
            output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        return hidden_states

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return self.embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )
