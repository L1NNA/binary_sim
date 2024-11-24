from torch.utils.data import Dataset
from tqdm import tqdm, trange
import json
import torch
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import random

from utils import get_tokens, BinFunc


class BinSimDataset(Dataset):

    def __init__(self, stage:str='train', max_lines=0, max_num=5000):
        json_path = f'./datasets/{stage}.jsonl'
        bins = defaultdict(list)
        self.max_lines = max_lines
        self.pairs = []
        with open(json_path) as f:
            i = 0
            for line in tqdm(f, desc=f'Loading {stage} dataset'):
                line = line.strip()
                js = BinFunc(**json.loads(line))
                self.pairs.append(js)
                bins[js.function].append(i)
                i += 1
        
        self.indices = []
        keys = [key for key in bins.keys() if len(bins[key]) >= 2]
        for _ in trange(max_num, desc=f'Sampling {stage} dataset'):
            key = random.choice(keys)
            neg_key = random.choice(list(bins.keys()))
            while neg_key == key:
                neg_key = random.choice(list(bins.keys()))
            index, pos_index = random.sample(range(len(bins[key])), 2)
            pos_index = bins[key][pos_index]
            index = bins[key].pop(index)
            neg_index = random.choice(bins[neg_key])
            self.indices.append((index, pos_index, 1))
            self.indices.append((index, neg_index, -1))
            if len(bins[key]) < 2:
                keys.remove(key)
            if len(keys) == 0:
                break

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x_index, y_index, label = self.indices[index]
        x_js = self.pairs[x_index]
        y_js = self.pairs[y_index]
        return x_js.get_blocks(self.max_lines), y_js.get_blocks(self.max_lines), label
    
    
def pairwise_collate(batch, tokenizer, max_blocks, max_length=64):
    x_batches, y_batches, labels = zip(*batch)
    
    x_inputs = get_tokens(x_batches, tokenizer, max_blocks, max_length)
    y_inputs = get_tokens(y_batches, tokenizer, max_blocks, max_length)

    result = {
        'input_ids': x_inputs['input_ids'],
        'attention_mask': x_inputs['attention_mask'],
        'y_input_ids': y_inputs['input_ids'],
        'y_attention_mask': y_inputs['attention_mask'],
        'labels': torch.tensor(labels, dtype=torch.long),
    }
    return result