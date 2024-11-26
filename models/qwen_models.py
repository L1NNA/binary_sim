import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from models.base_embedding_model import BaseModelForEmbedding

class Qwen2Model(BaseModelForEmbedding):
    
    def __init__(self, config:Qwen2Config):
        super(Qwen2Model, self).__init__(config)

    def get_model(self):
        return Qwen2Model(self.config)
    
    @classmethod
    def from_causal_lm(cls, causalLM:Qwen2ForCausalLM):
        config = causalLM.config
        model = cls(config)

        # be careful about the device:cuda,cpu or data parallel
        model.backbone.load_state_dict(causalLM.model.state_dict())
        return model