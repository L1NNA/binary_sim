from transformers import Qwen2Model, Qwen2ForCausalLM

from models.base_embedding_model import EmbeddingMixin


class Qwen2ForSequenceEmbedding(Qwen2Model, EmbeddingMixin):

    def get_hidden_state(self, input_ids, attention_mask):
        return super(Qwen2Model, self).forward(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return super(EmbeddingMixin, self).embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )
    
    @classmethod
    def from_causal_lm(cls, causalLM:Qwen2ForCausalLM):
        config = causalLM.config
        model = cls(config)
        # be careful about the device:cuda,cpu or data parallel
        model.load_state_dict(causalLM.model.state_dict())
        return model
    

class Qwen2CausalForSequenceEmbedding(Qwen2ForCausalLM, EmbeddingMixin):

    def __init__(self, config):
        super().__init__(config)
        self.lm_head.requires_grad_ = False

    def get_hidden_state(self, input_ids, attention_mask):
        hidden_states =  super(Qwen2ForCausalLM, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True 
        )
        return hidden_states[-1]

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return super(EmbeddingMixin, self).embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )