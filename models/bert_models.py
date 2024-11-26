from transformers import RobertaConfig

from models.base_embedding_model import BaseModelForEmbedding


class GraphCodeBERTEmbedding(BaseModelForEmbedding):

    def __init__(self, config:RobertaConfig):
        super(GraphCodeBERTEmbedding, self).__init__(config)

    def get_hidden_state(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)
    
    def get_pooling(self, hidden_state, _):
        return hidden_state