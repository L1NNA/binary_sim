import json
import argparse
from datetime import datetime
from os.path import join
import numpy as np
from sklearn.metrics import (accuracy_score, auc, precision_score,
                             recall_score, roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity
from cute_ranking.core import mean_reciprocal_rank, r_precision, average_precision, dcg_at_k, ndcg_at_k
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn import DataParallel
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
from torch.utils.data import Dataset, DataLoader, random_split

from models.coe_cos_sim import CoECosSim
from data_loaders.pos_neg_bin_sim_dataset import BinSimDataset, line_collate
from data_loaders.test_retrieval_dataset import RetrievalDataset
import os
from collections import defaultdict

def precision_at_k(r, k):
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    val = np.mean(r)
    return max(val, r_precision(r))


def evaluate_mrr_recall1(dataloader):
    mrr, recall1 = 0, 0
    for query_embedding, pool_embeddings, ground_truth_idx in dataloader:
        query_embedding = query_embedding.squeeze(0)
        similarities = F.cosine_similarity(query_embedding, pool_embeddings)
        sorted_indices = torch.argsort(similarities, descending=True)

        # Check if ground truth is the top result
        rank = (sorted_indices == ground_truth_idx).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1 / rank
        recall1 += 1 if rank == 1 else 0

    num_queries = len(dataloader)
    mrr /= num_queries
    recall1 /= num_queries
    return mrr, recall1
    
    
def find_common_elements(*lists):
    lists = lists[0]
    # Find intersection of all lists
    common_elements = set(lists[0])
    for lst in lists[1:]:
        common_elements &= set(lst)
    return list(common_elements)
    
    
def get_function_keys(path):
    sbins = defaultdict(list)
    functions = []
    with open(path) as f:
        functions = [json.loads(line) for line in f]
    for s in functions:
        sbins[s['function']] = s['blocks']
    return list(sbins.keys())


def evaluate(source=None, target=None, relevance=None, similarity=None):
    """
    source: matrix 
    target: matrix
    relevance: matrix of shape (len(target), len(source))
    Two options:
        provide: source, target, and relevance
        provide: similarity and relevance
    """
    if similarity is None:
        if isinstance(source, np.ndarray) or isinstance(source[0], np.ndarray):
            # if it is ndarray, we use cosine similarity to calcuate the score
            similarity = cosine_similarity(target, source)
            # of shape [len(target), len(source)]
        else:
            # otherwise we use bm25
            # bm25 = BM25Okapi(source)
            # similarity = [bm25.get_scores(q) for q in tqdm(target)]
            # of shape [len(target), len(source)]
            source = [' '.join([str(a) for a in s]) for s in source]
            target = [' '.join([str(a) for a in t]) for t in target]
            # print(target[0])
            vectorizer = TfidfVectorizer(use_idf=True)
            vectorizer.fit(source)
            x = vectorizer.transform(source)
            y = vectorizer.transform(target)
            similarity = cosine_similarity(x, y)

    # sorted relevance matrix by score, 1 indicates relevance
    results = []
    for s, r in tqdm(zip(similarity, relevance), total=len(relevance)):
        r = [1 if i in r else 0 for i in range(len(s))]
        # print(s)
        # print(r)
        sorted_r = [t for _, _, t in sorted(
            zip(s, range(len(r)), r), reverse=True)]
        # print(sorted_r)
        results.append(sorted_r)
    res = np.array(results)

    def _m(m, *args):
        return np.mean([m(r, *args) for r in res])

    return {
        'Mean Reciprocal Rank': mean_reciprocal_rank(res),
        'Precision': _m(r_precision),
        'Precision@1': _m(precision_at_k, 1),
        'Precision@5': _m(precision_at_k, 5),
        'Precision@10': _m(precision_at_k, 10),
        'Area Under PR Curve': _m(average_precision),
        'DCG@1': _m(dcg_at_k, 1),
        'DCG@5': _m(dcg_at_k, 5),
        'DCG@10': _m(dcg_at_k, 10),
        'NDCG@1': _m(ndcg_at_k, 1),
        'NDCG@5': _m(ndcg_at_k, 5),
        'NDCG@10': _m(ndcg_at_k, 10),
    }