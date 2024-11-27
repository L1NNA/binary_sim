from transformers import RobertaConfig, RobertaModel

from models.base_embedding_model import BaseModelForEmbedding


class GraphCodeBERTEmbedding(BaseModelForEmbedding):
    
    _no_split_modules = ["RobertaEmbeddings", "RobertaLayer"]
    config_class = RobertaConfig

    def __init__(self, config:RobertaConfig):
        super(GraphCodeBERTEmbedding, self).__init__(config)

    def get_hidden_state(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    
    def get_pooling(self, hidden_state, _):
        return hidden_state
    
    def get_model(self):
        return RobertaModel(self.config)