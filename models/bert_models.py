from transformers import RobertaModel

from models.base_embedding_model import EmbeddingMixin


class GraphCodeBERTForSequenceEmbedding(RobertaModel, EmbeddingMixin):

    def get_hidden_state(self, input_ids, attention_mask, **kwargs):
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
    
    def get_pooling(self, hidden_state, _, **kwargs):
        return hidden_state

    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        return self.embedding(
            input_ids, attention_mask, y_input_ids, y_attention_mask, labels
        )