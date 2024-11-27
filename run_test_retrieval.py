import json
import argparse
from datetime import datetime
from os.path import join, exists
import os
import random
import pickle
import itertools

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from data_loaders.test_retrieval_dataset import RetrievalDataset
from utils.retrieval_utils import find_common_elements, get_detailed_instruct, test_retrieval
from models.llm2vec import Qwen2MNTPForSequenceEmbedding
from utils import get_tokens
from models.qwen_models import Qwen2ForSequenceEmbedding
from models.codet5p_models import CodeT5PEncoderForSequenceEmbedding
from models.bert_models import GraphCodeBERTForSequenceEmbedding

models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2ForSequenceEmbedding),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEncoderForSequenceEmbedding),
    'jina_emb': ('jinaai/jina-embeddings-v2-base-code', None),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTForSequenceEmbedding),
    'qwen_llm2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding)
}

def load_data(data_path, source, target, pool_size, max_lines):
    source_path = join(data_path, f'test_{source}.jsonl')
    target_path = join(data_path, f'test_{target}.jsonl')
    source_dataset = RetrievalDataset(source_path, keys=None)
    target_dataset = RetrievalDataset(target_path, keys=None)

    os.makedirs(join(data_path, 'samples'), exist_ok=True)
    cache_file = join(data_path, 'samples', f'{source}_to_{target}_{pool_size}.pt')
    # load pairs, the sampled pairs should be fixed across all baseline and models
    if exists(cache_file):
        with open(cache_file, 'rb') as f:
            sampled_keys = pickle.load(f)
    else:
        source_keys = list(source_dataset.bins['function'])
        target_keys = list(target_dataset.bins['function'])
        common_keys = find_common_elements([source_keys, target_keys])
        assert len(common_keys) >= pool_size, "pool size too large, not enough functions with this setup"
        sampled_keys = random.sample(common_keys, pool_size)
        with open(cache_file, 'wb') as f:
            pickle.dump(sampled_keys, f)

    source_funcs = source_dataset.get_all(keys=sampled_keys, join_blocks=(max_lines==0))
    target_funcs = target_dataset.get_all(keys=sampled_keys, join_blocks=(max_lines==0))
    return source_funcs, target_funcs

def get_embeddings(
    model, tokenizer,
    data_path, source, target, pool_size, max_length, max_blocks = 0,
    test_batch_size = 32,
    query_instruction = None,
    value_instruction = None,
    device = 'cuda:0'
):
    queries, values = load_data(data_path, source, target, pool_size, max_blocks)

    queries = [get_detailed_instruct(query_instruction, query) for query in queries]
    values = [get_detailed_instruct(value_instruction, value) for value in values]
    
    query_embs = []
    value_embs = []
    for i in trange(0, len(queries), test_batch_size, desc="Embedding queries"):
        with torch.no_grad():
            batch_dict = get_tokens(queries[i:i+test_batch_size], tokenizer, max_blocks, max_length).to(device)
            query_outputs = model(**batch_dict).embedding.cpu().float()
            query_embs.append(query_outputs)
    query_embs = torch.cat(query_embs, dim=0).view(-1, query_outputs.size(-1))

    for i in trange(0, len(values), test_batch_size, desc="Embedding values"):
        with torch.no_grad():
            batch_dict = get_tokens(values[i:i+test_batch_size], tokenizer, max_blocks, max_length).to(device)
            value_outputs = model(**batch_dict).embedding.cpu().float()
            value_embs.append(value_outputs)
    value_embs = torch.cat(value_embs, dim=0).view(-1, value_outputs.size(-1))

    metrics = test_retrieval(query_embs, value_embs)
    return metrics

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # === model ===
    parser.add_argument("--model", choices=models.keys(), default='jina_emb', help="The model name")
    parser.add_argument("--local_model_path", type=str,)
    parser.add_argument("--local_tokenizer_path", type=str)
    parser.add_argument(
        # "--sft", type=str, default='lora',
        "--sft_model", type=str, default=None,
        help="whether to use sft model or pre-trained model"
    )

    # === data ===
    parser.add_argument(
        "--test_batch_size", type=int, default=32,
        help="The batch size."
    )
    parser.add_argument(
        "--max_blocks", type=int, default=0,
        help="max number of blocks"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="max number of tokens per line"
    )
    parser.add_argument(
        "--pool_size", type=int, default=1000,
        help="pool size for retrieval"
    )
    parser.add_argument(
        "--data_path", type=str, default=join(os.getcwd(), 'datasets'),
        help="path to the datasets"
    )
    # Parse the arguments
    args = parser.parse_args()
    

    # ================= Load Model ======================
    model_path, model_cls = models[args.model]
    model = model_cls.from_pretrained(
        model_path if args.local_model_path is None else args.local_model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if args.local_tokenizer_path is None else args.local_tokenizer_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    # ================= Embedding ======================
    optimizations = ['o0', 'o1', 'o2', 'o3']
    obfuscations = ['obf_all', 'obf_none', 'obf_sub', 'obf_fla', 'obf_bcf']
    compilers = ['clang', 'gcc']
    architectures = ['arm', 'powerpc', 'x86_32', 'x86_64', 'mips']
    
    os.makedirs(join('./results', 'reterival'), exist_ok=True)
    result_file = join('./results', 'reterival', f'{args.model}_{args.pool_size}.csv')
    with open(result_file, 'w') as f:
        f.write('source, dest, mrr, recall@1, recall@10\n')

    combinations = itertools.permutations(optimizations, 2)
    for o1, o2 in combinations:
        print(f'Testing reterival optimization {o1} to optimization {o2}')
        metrics = get_embeddings(
            model, tokenizer,
            args.data_path, o1, o2, args.pool_size, args.max_length, args.max_blocks,
            args.test_batch_size, f'translate the following binary code in optimization {o1} to optimization {o2}',
        )
        print(f'Finished testing reterival optimization {o1} to optimization {o2}', metrics)
        with open(result_file, 'a') as f:
            f.write(f"{o1}, {o2}, {metrics['mrr']}, {metrics['recall_at_1']}, {metrics['recall_at_10']}\n")

    combinations = itertools.permutations(obfuscations, 2)
    for o1, o2 in combinations:
        print(f'Testing reterival obfuscation {o1} to obfuscation {o2}')
        metrics = get_embeddings(
            model, tokenizer,
            args.data_path, o1, o2, args.pool_size, args.max_length, args.max_blocks,
            args.test_batch_size, f'translate the following binary code obfuscated by {o1} to obfuscation {o2}',
        )
        print(f'Finished testing reterival obfuscation {o1} to obfuscation {o2}', metrics)
        with open(result_file, 'a') as f:
            f.write(f"{o1}, {o2}, {metrics['mrr']}, {metrics['recall_at_1']}, {metrics['recall_at_10']}\n")

    combinations = itertools.permutations(compilers, 2)
    for c1, c2 in combinations:
        print(f'Testing reterival compiler {c1} to compiler {c2}')
        metrics = get_embeddings(
            model, tokenizer,
            args.data_path, c1, c2, args.pool_size, args.max_length, args.max_blocks,
            args.test_batch_size, f'translate the following binary code compiled by {c1} to compiler {c2}',
        )
        print(f'Finished testing reterival compiler {c1} to compiler {c2}', metrics)
        with open(result_file, 'a') as f:
            f.write(f"{c1}, {c2}, {metrics['mrr']}, {metrics['recall_at_1']}, {metrics['recall_at_10']}\n")

    combinations = itertools.permutations(architectures, 2)
    for a1, a2 in combinations:
        print(f'Testing reterival architecture {a1} to architecture {a2}')
        metrics = get_embeddings(
            model, tokenizer,
            args.data_path, a1, a2, args.pool_size, args.max_length, args.max_blocks,
            args.test_batch_size, f'translate the following binary code in architecture {a1} to architecture {a2}',
        )
        print(f'Finished testing reterival architecture {a1} to architecture {a2}', metrics)
        with open(result_file, 'a') as f:
            f.write(f"{a1}, {a2}, {metrics['mrr']}, {metrics['recall_at_1']}, {metrics['recall_at_10']}\n")
    
    