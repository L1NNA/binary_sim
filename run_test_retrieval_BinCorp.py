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
from models.qwen_models import Qwen2ForSequenceEmbedding, CustomQwen2ForSequenceEmbedding
from models.codet5p_models import CodeT5PEncoderForSequenceEmbedding
from models.bert_models import GraphCodeBERTForSequenceEmbedding

models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2ForSequenceEmbedding),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEncoderForSequenceEmbedding),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTForSequenceEmbedding),
    'qwen_llm2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding),
    'uptrainedcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2ForSequenceEmbedding),
    'customcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2ForSequenceEmbedding),
}

### python run_test_retrieval_BinCorp.py --model='customcodeqwen2vec' --local_model_path='model_checkpoints/simcse_customcodeqwen2vec_BinaryCorp/checkpoint-10000'

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

    source_funcs = source_dataset.get_all(keys=sampled_keys)
    target_funcs = target_dataset.get_all(keys=sampled_keys)
    return source_funcs, target_funcs

def get_embeddings(
    model, tokenizer,
    data_path, source, target, pool_size, max_length, max_blocks = 0,
    test_batch_size = 32,
    query_instruction = None,
    value_instruction = None,
    get_blk_mask=False,
    get_arch=False,
    device = 'cuda:0'
):
    queries, values = load_data(data_path, source, target, pool_size, max_blocks)

    queries = [get_detailed_instruct(query_instruction, query) for query in queries]
    values = [get_detailed_instruct(value_instruction, value) for value in values]
    
    query_embs = []
    value_embs = []
    for i in trange(0, len(queries), test_batch_size, desc="Embedding queries"):
        with torch.no_grad():
            batch_dict = get_tokens(queries[i:i+test_batch_size], tokenizer, max_blocks, max_length, get_blk_mask=get_blk_mask, get_arch=get_arch).to(device)
            query_outputs = model(**batch_dict).embedding.cpu().float()
            query_embs.append(query_outputs)
    query_embs = torch.cat(query_embs, dim=0).view(-1, query_outputs.size(-1))

    for i in trange(0, len(values), test_batch_size, desc="Embedding values"):
        with torch.no_grad():
            batch_dict = get_tokens(values[i:i+test_batch_size], tokenizer, max_blocks, max_length, get_blk_mask=get_blk_mask, get_arch=get_arch).to(device)
            value_outputs = model(**batch_dict).embedding.cpu().float()
            value_embs.append(value_outputs)
    value_embs = torch.cat(value_embs, dim=0).view(-1, value_outputs.size(-1))

    metrics = test_retrieval(query_embs, value_embs)
    return metrics

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    
    # === hyperparameters ===
    parser.add_argument("--instruct", type=int, default=0)

    # === model ===
    parser.add_argument("--model", choices=models.keys(), default='uptrainedcodeqwen2vec', help="The model name")
    parser.add_argument("--local_model_path", type=str,)
    parser.add_argument("--local_tokenizer_path", type=str)
    parser.add_argument("--note", type=str)
    parser.add_argument("--prompt", action="store_true", help="use prompt")

    # === data ===
    parser.add_argument(
        "--test_batch_size", type=int, default=64,
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
        "--pool_size", type=int, default=10000,
        help="pool size for retrieval"
    )
    parser.add_argument(
        "--data_path", type=str, default=join(os.getcwd(), 'datasets'),
        help="path to the datasets"
    )
    parser.add_argument(
        "--get_blk_mask", action='store_true',
        help="use blk mask in input"
    )
    parser.add_argument(
        "--get_arch", action='store_true',
        help="use arch embedding"
    )
    parser.add_argument(
        "--block_gate", action='store_true',
        help="whether to use block gate or block bias, leave empty and enable --require_blk_mask for blk_bias, else enable both for adding block gate parameters"
    )
    parser.add_argument("--attention", type=str, help="attention type, leave empty for default sdpa or eager for custom")
    parser.add_argument("--use_flex", action="store_true", help="use flex attention")
    # Parse the arguments
    args = parser.parse_args()

    # ================= Load Model ======================
    model_path, model_cls = models[args.model]
    config = AutoModel.from_pretrained(model_path).config
    if model_cls is CustomQwen2ForSequenceEmbedding:
        config._attn_implementation = 'eager'
    if args.use_flex:
        config.use_flex = True
    if args.get_arch:
        config.arch_embed = True
    if args.get_blk_mask:
        config.require_blk_mask = True
        if args.block_gate:
            config.block_gate = True
    if args.block_gate and not args.get_blk_mask:
        raise "require_blk_mask must be enabled to use block_gate"

    model = model_cls.from_pretrained(
        model_path if args.local_model_path is None else args.local_model_path,
        # device_map='auto',
        torch_dtype=torch.bfloat16,
        config=config,
    ).to('cuda:0')
    model = nn.DataParallel(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if args.local_tokenizer_path is None else args.local_tokenizer_path,
        trust_remote_code=True,
        padding_side='left',
        truncation_side='left',
    )

    special_tokens = ['<addr>', '<byte>', '<str>', '<BLK>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.eos_token = '<EOS>'
    tokenizer.sep_token = '<SEP>'
    tokenizer.pad_token = '<PAD>'
    tokenizer.unk_token = '<unk>'
    
    # ================= Embedding ======================
    optimizations = ['o0_BinaryCorp', 'o1_BinaryCorp', 'o2_BinaryCorp', 'o3_BinaryCorp', 'os_BinaryCorp']
    
    os.makedirs(join('./results', 'retrieval_BinaryCorp'), exist_ok=True)
    result_file = join('./results', 'retrieval_BinaryCorp', f'{args.model}-{args.instruct}_{args.pool_size}_{args.note}.csv')
    with open(result_file, 'w') as f:
        f.write('source, dest, mrr, recall@1, recall@10\n')

    combinations = itertools.combinations(optimizations, 2)
    for o1, o2 in combinations:
        instruction = None
        if args.instruct == 1:
            instruction = f'provide the semantics for the following assembly function, which is compiled with the setting: '
        print(f'Testing reterival optimization {o1} to optimization {o2}')
        metrics = get_embeddings(
            model, tokenizer,
            args.data_path, o1, o2, args.pool_size, args.max_length, args.max_blocks,
            args.test_batch_size, instruction,
            get_blk_mask=args.get_blk_mask,
            get_arch=args.get_arch,
        )
        print(f'Finished testing reterival optimization {o1} to optimization {o2}', metrics)
        with open(result_file, 'a') as f:
            f.write(f"{o1}, {o2}, {metrics['mrr']}, {metrics['recall_at_1']}, {metrics['recall_at_10']}\n")

    