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


def plot_t_SNE(embeddings, func_labels, trans_labels, func_names, trans):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(trans), learning_rate='auto', init='random')
    X_tsne = tsne.fit_transform(X_scaled)

    # Plotting
    plt.figure(figsize=(10, 8))

    # create len(func_names) colors
    colors = plt.cm.tab10.colors
    # create len(trans) shapes
    shapes = ['o', 's', '^', 'v', 'D']
    assert len(shapes) >= len(trans), f"Not enough number of shapes {len(trans)}, only support {len(shapes)}"
    
    for i in range(embeddings.shape[0]):
        color = colors[func_labels[i]]
        shape = shapes[trans_labels[i]]
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=color, marker=shape, alpha=0.6)

    # create a standalone legend for the shapes
    for i in range(len(trans)):
        mask = (y == 0) & (z == i)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[0], marker=shapes[i], alpha=0.6, label=trans[i])

    plt.title('t-SNE Visualization')
    plt.legend()
    plt.show()
    
    
models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2ForSequenceEmbedding),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEncoderForSequenceEmbedding),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTForSequenceEmbedding),
    'qwen_llm2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding),
    'uptrainedcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2ForSequenceEmbedding),
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
    return funcs, sampled_keys, sources


def get_embeddings(
    model, tokenizer,
    data_path, sources, max_length,
    pool_size=100, max_blocks = 0,
    test_batch_size = 32,
    device = 'cuda:0'
):
    func_groups, sampled_keys, sources = load_data(data_path, sources, pool_size)
    
    embeddings = []
    func_labels, trans_labels = [], []
    
    for i,source in enumerate(sources):
        funcs = func_groups[i]
        for j in trange(0, len(funcs), test_batch_size, desc=f"Embedding {source}"):
            with torch.no_grad():
                batch_dict = get_tokens(funcs[j:j+test_batch_size], tokenizer, max_blocks, max_length).to(device)
                query_outputs = model(**batch_dict).embedding.cpu().float()
                embeddings.append(query_outputs)
    embeddings = torch.cat(query_embs, dim=0).numpy()
    return embeddings, func_labels, trans_labels