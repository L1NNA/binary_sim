import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

from peft import LoraConfig, get_peft_model

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")

def reload_module(m):
    import importlib
    importlib.reload(m)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(pred, output_file=None):
    preds = pred.predictions.argmax(-1)  # Get the predicted class by selecting the max logit
    labels = pred.label_ids
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    metrics = {
        'accuracy': acc,
        'f1':f1,
        'recall':recall,
        'precision':precision
    }
    if output_file is not None:
        with open(output_file, 'a') as of:
            of.write(json.dumps(metrics)+'\n')
    return metrics

def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    model.config.gradient_checkpointing = True
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model

def preserve_middle_section(original_list, max_length):
    length = len(original_list)
    middle = length / 2
    if max_length % 2 == 0:
        start = int(middle - max_length / 2)
        end = int(middle + max_length / 2)
    else:
        start = int(middle - (max_length // 2))
        end = int(middle + (max_length // 2) + 1)
    return original_list[start:end]

def get_tokens(batches, tokenizer, max_blocks, max_length):
    b = len(batches)
    
    ins = []
    if max_blocks == 0:
        ins.extend(batches)  
    else:
        # TODO: get local maximum
        for blocks in batches:
            if max_blocks > len(blocks):
                ins.extend([""]*(max_blocks - len(blocks)))
                ins.extend(blocks)  
            elif max_blocks <= len(blocks):
                ins.extend(preserve_middle_section(blocks, max_blocks))

    inputs = tokenizer(
        ins, return_tensors='pt',
        padding=True, truncation=True, max_length=max_length
    )
    
    if max_blocks > 0:
        inputs['input_ids'] = inputs['input_ids'].reshape(b, max_blocks, -1)
        inputs['attention_mask'] = inputs['attention_mask'].reshape(b, max_blocks, -1)
    return inputs

@dataclass
class BinFunc:
    function: str
    blocks: List[str]
    file: str
    
    def get_blocks(self, max_blocks=0, use_blk_token = True, remove_last=False) -> str:
        blocks = self.blocks
        if remove_last:
            blocks = blocks[:-1]
        # move the tokens containing arch, opt, compiler, etc.. to the beginning for target sentence in paired causal training 
        if max_blocks == 0:
            if use_blk_token:
                return ' <BLK> '.join(blocks)
            else:
                return ' '.join(blocks)
        if len(blocks) > max_blocks:
            remant = (len(blocks) - max_blocks) // 2
            blocks = blocks[remant:remant+max_blocks]
        return blocks