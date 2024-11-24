from typing import List

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import find_common_elements, calculate_mrr, compute_retrieval_metrics, get_tokens


def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

def get_embeddings(
    queries:List[str], values:List[str], model, tokenizer, max_length, test_batch_size = 32,
    max_blocks = 0,
    task_source = 'Given an assembly function, retrieve the clone',
    task_target = 'Given an assembly function, retrieve the clone',
    device = 'cuda:0',
    sft = None
):
    query_embs = []
    value_embs = []
    
    if max_blocks == 0 and task_source != None and task_target != None:
        queries = [get_detailed_instruct(task_source, query) for query in queries]
        values = [get_detailed_instruct(task_target, value) for value in values]
    
    for i in trange(0, len(queries), test_batch_size, desc="Embedding queries"):
        with torch.no_grad():
            batch_dict = get_tokens(queries[i:i+test_batch_size], tokenizer, max_blocks, max_length).to(device)
            if sft is not None:
                batch_dict['sft'] = sft
            query_outputs = model(**batch_dict)['preds'].cpu()
            query_embs.append(query_outputs)
    query_embs = torch.cat(query_embs, dim=0).view(-1, query_outputs.size(-1))

    for i in trange(0, len(values), test_batch_size, desc="Embedding values"):
        with torch.no_grad():
            batch_dict = get_tokens(values[i:i+test_batch_size], tokenizer, max_blocks, max_length).to(device)
            if sft is not None:
                batch_dict['sft'] = sft
            value_outputs = model(**batch_dict)['preds'].cpu()
            value_embs.append(value_outputs)
    value_embs = torch.cat(value_embs, dim=0).view(-1, value_outputs.size(-1))

    return query_embs, value_embs
