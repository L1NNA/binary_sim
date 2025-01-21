# from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import pickle
import re
import torch
import gc
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import gzip
import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import random
from collections import defaultdict
from ast import literal_eval
from data_loaders.test_retrieval_dataset import RetrievalDataset
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from utils import reload_module, get_tokens
from data_loaders.pos_neg_bin_sim_dataset import BinSimDataset, pairwise_collate
from data_loaders.causal_dataset import CausalDataset, causal_collate, CausalDatasetPair
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from utils import get_tokens, BinFunc
from utils.retrieval_utils import find_common_elements, get_detailed_instruct, test_retrieval
from dataclasses import dataclass
from typing import Optional, List, Dict
from tokenizers import SentencePieceBPETokenizer
from models.causalLM import CustomModelCausal, CustomConfig
from peft import get_peft_model, LoraConfig
from transformers import Cache, DynamicCache

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device='cpu'

def main():
    models = {
        'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct'),
        'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding'),
        'jina_emb': ('jinaai/jina-embeddings-v2-base-code'),
        'codebert': ('microsoft/codebert-base'),
        'qwen_llm2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct'),
        'qwen_sft': ('Qwen/Qwen2.5-Coder-0.5B-Instruct')
    }

    model_path = models['codet5p-110m-embedding']

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left',
        truncation_side='left'
    )

    # special_tokens = ['<addr>', '<byte>', '<str>', '<BLK>']
    # tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    # tokenizer.sep_token = '<SEP>'
    # tokenizer.unk_token = '<unk>'




    pool_size=1000
    path1 = os.path.join(os.getcwd(), 'datasets', 'test_o0.jsonl')
    path2 = os.path.join(os.getcwd(), 'datasets', 'test_o1.jsonl')
    source = RetrievalDataset(path1)
    target = RetrievalDataset(path2)
    source_keys = list(source.bins['function'])
    target_keys = list(target.bins['function'])
    common_keys = find_common_elements([source_keys, target_keys])
    assert len(common_keys) >= pool_size, "pool size too large, not enough functions with this setup"
    sampled_keys = random.sample(common_keys, pool_size) 

    source_funcs = source.get_all(keys=sampled_keys)
    target_funcs = target.get_all(keys=sampled_keys)

    with open('o0_o1_generated_tokens.jsonl', 'r') as f:
        token_output = [json.loads(line) for line in f]
    token_output = token_output[0]
    common_keys = token_output.keys()
    target_funcs = target.get_all(keys=sampled_keys)
    token_output = list(token_output.values())

    token_output = [i.replace('<PAD>', tokenizer.pad_token) for i in token_output]

    query_embs = []
    value_embs = []
    test_batch_size=32
    for i in trange(0, len(token_output), test_batch_size, desc="Embedding queries"):
        with torch.no_grad():
            batch_dict = tokenizer(token_output[i:i+test_batch_size], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
            query_outputs = model(**batch_dict)['preds'].cpu()
            query_embs.append(query_outputs)
    query_embs = torch.cat(query_embs, dim=0).view(-1, query_outputs.size(-1))

    for i in trange(0, len(target_funcs), test_batch_size, desc="Embedding values"):
        with torch.no_grad():
            batch_dict = tokenizer(target_funcs[i:i+test_batch_size], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
            value_outputs = model(**batch_dict)['preds'].cpu()
            value_embs.append(value_outputs)
    value_embs = torch.cat(value_embs, dim=0).view(-1, value_outputs.size(-1))

    metrics = test_retrieval(query_embs, value_embs)
    print(metrics)


if __name__ == "__main__":
    main()

