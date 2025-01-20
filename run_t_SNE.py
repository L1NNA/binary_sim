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
from collections import defaultdict
from collections import namedtuple
from typing import List

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

selected_colors = ["582f0e","7f4f24","936639","a68a64","b6ad90","c2c5aa","a4ac86","656d4a","414833","333d29"] + \
         ["03045e","023e8a","0077b6","0096c7","00b4d8","48cae4","90e0ef","ade8f4","caf0f8"] + \
         ["590d22","800f2f","a4133c","c9184a","ff4d6d","ff758f","ff8fa3","ffb3c1","ffccd5","fff0f3"] + \
         ["0466c8","0353a4","023e7d","002855","001845","001233","33415c","5c677d","7d8597","979dac"]



def train_t_SNE(embeddings, perplexity=30, random_state=0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, learning_rate='auto', init='random')
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne


def plot_t_SNE(X_tsne, embeddings, func_labels, opt_labels, func_names, opts, title, output_path):
    # Plotting
    plt.figure(figsize=(10, 8))

    # create len(func_names) colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(func_names)))
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
    # for i in range(len(func_names)):
    #     mask = (func_labels == i) & (opt_labels == 0)
    #     plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[i], marker='o', alpha=0.6, label=func_names[i])

    plt.title(title)
    plt.legend()
    

    # save the plot
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    
Embeddings = namedtuple('Embeddings', ['embeddings', 'model_name', 'source', 'func_names'])

models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2ForSequenceEmbedding),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEncoderForSequenceEmbedding),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTForSequenceEmbedding),
    'qwen_llm2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding),
    'customcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2ForSequenceEmbedding), # CustomQwen2ForSequenceEmbedding
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
    # embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def gen_main():
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
        'customcodeqwen2vec': './model_checkpoints/simcse_customcodeqwen2vec_uptrained/checkpoint-34688',
        'qwen_emb': './model_checkpoints/simcse_qwen_emb/checkpoint-15000',
        'codet5p-110m-embedding': './model_checkpoints/simcse_codet5p-110m-embedding/checkpoint-1608',
        'graphcodebert': './model_checkpoints/simcse_graphcodebert/checkpoint-2345',
        'qwen_llm2vec': './model_checkpoints/simcse_codeqwen2vec/checkpoint-11250',
    }
    
    model_checkpoints = {
        'qwen_emb': None,
        'qwen_llm2vec': None,
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
            checkpoint if checkpoint else model_path,
            # device_map='auto',
            torch_dtype=torch.bfloat16
        ).to(args.device)
        model = nn.DataParallel(model)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left',
            truncation_side='left',
        )

        if 'coder' in model_path:
            special_tokens = ['<addr>', '<byte>', '<str>', '<BLK>']
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            tokenizer.eos_token = '<EOS>'
            tokenizer.sep_token = '<SEP>'
            tokenizer.pad_token = '<PAD>'
            tokenizer.unk_token = '<unk>'

        
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
                    'model_name': model_name + ('_zero' if checkpoint is None else ''),
                    'source': source,
                    'embeddings': embeddings,
                    'func_names': func_names
                }) + '\n')
                
                
def plot_main(sources_group=[
        ['arm', 'powerpc', 'x86_32', 'x86_64', 'mips'],
        ['o0','o1','o2','o3'],
        ['obf_all','obf_none','obf_sub','obf_fla','obf_bcf'],
        ['clang','gcc'],
        ['arm','powerpc','x86_32','x86_64','mips'],
    ]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='output jsonl path')
    parser.add_argument('--num_of_funcs', type=int, default='output jsonl path')
    parser.add_argument('--source', type=int, default='output jsonl path')
    args = parser.parse_args()
    
    results:List[Embeddings] = []

    with open(args.input_path, 'r') as f:
        for line in f:
            results.append(Embeddings(**json.loads(line)))
    
    num_of_funcs = args.num_of_funcs
    num_of_sources = args.num_of_sources

    func_names = results[0].func_names[:num_of_funcs]
    sources = sources_group[args.source)
    func_labels = np.concatenate([
        [*range(len(func_names))] * len(sources)
    ], axis=0)
    source_labels = np.concatenate([
        [i] * len(func_names) for i in range(len(sources))
    ], axis=0)
    
    model_embeddings = defaultdict(dict)
    for result in results:
        model_embeddings[result.model_name][result.source] = result.embeddings
    
    for model_name in model_embeddings:
  
        all_embs = np.concatenate(
            [model_embeddings[model_name][key][:num_of_funcs] for key in sources],
            axis=0
        )
        tsne = train_t_SNE(qwen_llm2vec_embs, perplexity=num_of_funcs, random_state=0)
        plot_t_SNE(tsne, qwen_llm2vec_embs, func_labels, source_labels, func_names, sources, model_name)
    

if __name__ == '__main__':
    plot_main()