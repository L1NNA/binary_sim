import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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


def train_t_SNE(embeddings, perplexity=30, random_state=0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, learning_rate='auto', init='random')
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne

def plot_t_SNE(X_tsne, embeddings, func_labels, opt_labels, func_names, opts):
    # Plotting
    plt.figure(figsize=(10, 8))

    # create len(func_names) colors
    colors = plt.cm.tab10.colors
    # create len(trans) shapes
    shapes = ['o', 's', '^', 'v', 'D']
    assert len(shapes) >= len(opts), f"Not enough number of shapes {len(opts)}, only support {len(shapes)}"
    
    for i in range(embeddings.shape[0]):
        color = colors[func_labels[i]]
        shape = shapes[opt_labels[i]]
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=color, marker=shape, alpha=0.6)

    # create a standalone legend for the shapes
    for i in range(len(opts)):
        mask = (func_labels == 0) & (opt_labels == i)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[0], marker=shapes[i], alpha=0.6, label=opts[i])

    # create a standalone legend for the colors
    for i in range(len(func_names)):
        mask = (func_labels == i) & (opt_labels == 0)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[i], marker='o', alpha=0.6, label=func_names[i])

    plt.title('t-SNE Visualization')
    plt.legend()
    plt.show()
    
    
models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2ForSequenceEmbedding),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEncoderForSequenceEmbedding),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTForSequenceEmbedding),
    'qwen_llm2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding),
    'customcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2ForSequenceEmbedding),
}

def load_data(data_path, sources, pool_size=100):
    
    datasets = [
        RetrievalDataset(join(data_path, f'test_{source}.jsonl'), keys=None) \
        for source in sources
    ]
    keys = [dataset.bins['function'] for dataset in datasets]
    common_keys = find_common_elements(keys)
    sampled_keys = random.sample(common_keys, pool_size)
    funcs = [dataset.get_all(keys=sampled_keys) for dataset in datasets]
    return funcs, sampled_keys


def get_embeddings(
    model_name, model, tokenizer,
    source, funcs,
    max_blocks, max_length, test_batch_size = 32,
    device = 'cuda:0'
):
    
    embeddings = []
    for j in trange(0, len(funcs), test_batch_size, desc=f"Embedding {source} by {model_name}"):
        with torch.no_grad():
            batch_dict = get_tokens(funcs[j:j+test_batch_size], tokenizer, max_blocks, max_length).to(device)
            query_outputs = model(**batch_dict).embedding.cpu().float().numpy().tolist()
            embeddings.extend(query_outputs)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--max_blocks', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_path', type=str, default='output jsonl path')
    args = parser.parse_args()

    model_checkpoints = {
        'qwen_emb': './model_checkpoints/simcse_qwen_emb/checkpoint-15000',
        'codet5p-110m-embedding': './model_checkpoints/simcse_codet5p-110m-embedding/checkpoint-1608',
        'graphcodebert': './model_checkpoints/simcse_graphcodebert/checkpoint-2345',
        'qwen_llm2vec': './model_checkpoints/simcse_codeqwen2vec_uptrained/checkpoint-67500',
        'customcodeqwen2vec': './model_checkpoints/simcse_customcodeqwen2vec_uptrained/checkpoint-34688',
    }
    
    sources = [
        'o0', 'o1', 'o2', 'o3',
        'obf_all', 'obf_none', 'obf_sub', 'obf_fla', 'obf_bcf',
        'clang', 'gcc',
        'arm', 'powerpc', 'x86_32', 'x86_64', 'mips'
    ]

    func_group, func_names = load_data(args.data_path, sources, args.pool_size)

    for model_name, checkpoint in model_checkpoints.items():
        if model_name not in models:
            continue
    
        model_path, model_class = models[model_name]
        model = model_class.from_pretrained(
            checkpoint,
            # device_map='auto',
            torch_dtype=torch.bfloat16
        ).to(args.device)
        model = nn.DataParallel(model)
    
    
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
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
        model.eval()

        
        for i in range(len(sources)):
            funcs = func_group[i]
            source = sources[i]
            embeddings = get_embeddings(
                model_name, model, tokenizer,
                source, funcs,
                args.max_blocks, args.max_length, args.test_batch_size,
                args.device
            )

            with open(args.output_path, 'a') as f:
                f.write(json.dumps({
                    'model_name': model_name,
                    'source': source,
                    'embeddings': embeddings.tolist(),
                    'func_names': func_names
                }) + '\n')