from typing import List

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_detailed_instruct(task_description: str, query: str) -> str:
        if task_description is None:
            return query
        return f'Instruct: {task_description}\nQuery: {query}'

def calculate_mrr(scores:np.ndarray, relevance:np.ndarray):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a batch of data where each row contains relevance scores.
    
    Args:
    scores (np.ndarray): An array of shape (query_batch, values_batch) where the relevant item for each query is at the diagonal position (i, i).
    relevance (np.ndarray): skip ith element if the ith masking is zero.
    
    Returns:
    float: The Mean Reciprocal Rank (MRR).
    """
    query_batch = scores.shape[0]
    
    # Initialize a list to store ranks
    ranks = []

    for i in range(query_batch):
            
        # Get the relevance scores for the i-th row
        relevance_scores = scores[i]
        relevant_score = relevance_scores[relevance[i]]
        
        # Calculate the rank of the relevant score within the row
        rank = (relevance_scores > relevant_score).sum().item() + 1
        
        # Append the rank to the list
        ranks.append(rank)
    
    # Convert ranks to an array
    ranks = np.array(ranks)
    
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1 / ranks
    
    # Compute the Mean Reciprocal Rank (MRR)
    mrr = reciprocal_ranks.mean()
    
    return float(mrr)


def find_common_elements(*lists):
    lists = lists[0]
    # Find intersection of all lists
    common_elements = set(lists[0])
    for lst in lists[1:]:
        common_elements &= set(lst)
    return list(common_elements)


def recall_at_k(scores: np.ndarray, relevance:np.ndarray, k: int) -> float:
    """
    Compute the recall@k for a scores matrix.
    
    Parameters:
    - scores: np.ndarray of shape (query_batch_size, values_batch_size)
    - relevance: np.ndarray of shape (query_batch_size)
    - k: int, the number of top items to consider
    
    Returns:
    - float, the average recall@k
    """
    # Get the batch size
    query_batch = scores.shape[0]
    
    # Get the indices of the top-k scores for each query
    top_k_indices = np.argsort(-scores, axis=1)[:, :k]
    
    # Initialize the recall@k counter
    recall_at_k_count = 0
    
    # Check if the relevant item (diagonal element) is among the top-k items
    for i in range(query_batch):
        if relevance[i] in top_k_indices[i]:
            recall_at_k_count += 1
    
    # Compute the average recall@k
    recall_at_k = recall_at_k_count / query_batch
    
    return recall_at_k


def test_retrieval(query_embs, value_embs):
    scores = []
    for i in query_embs:
        scores.append(F.cosine_similarity(i.unsqueeze(0), value_embs, dim=1).numpy())
    scores = np.array(scores)
    
    ## this takes too much memory for large pool size like 10k
    # scores = F.cosine_similarity(query_embs.unsqueeze(1), value_embs.unsqueeze(0), dim=2).numpy()
    relevance = np.arange(query_embs.size(0))
    return compute_retrieval_metrics(scores, relevance)

def compute_retrieval_metrics(scores, relevance):
    pool_size = scores.shape[1]
    if relevance is None:
        relevance = np.arange(scores.shape[0])
    mrr = calculate_mrr(scores, relevance)
    recall_at_1 = recall_at_k(scores, relevance, 1)
    recall_at_10 = recall_at_k(scores, relevance, 10)
    return {
        'mrr':mrr,
        'recall_at_1':recall_at_1,
        'recall_at_10':recall_at_10,
        'pool_size': pool_size
    }