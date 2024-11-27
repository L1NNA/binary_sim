from transformers import RobertaModel

from models.base_embedding_model import EmbeddingMixin


class GraphCodeBERTForSequenceEmbedding(RobertaModel, EmbeddingMixin):

    def get_hidden_state(self, input_ids, attention_mask):
        return super(RobertaModel, self).forward(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
    
    def get_pooling(self, hidden_state, _):
        return hidden_state

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return super(EmbeddingMixin, self).embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )