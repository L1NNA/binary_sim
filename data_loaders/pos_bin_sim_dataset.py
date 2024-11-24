from torch.utils.data import Dataset
from tqdm import tqdm, trange
import json
import torch
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import random

from utils import get_tokens, BinFunc


class PosBinSimDataset(Dataset):

    def __init__(self, stage:str='train', max_lines=0, max_num=10000):
        self.max_lines = max_lines
        self.pairs = []
        
        json_path = f'./datasets/{stage}.jsonl'
        bins = defaultdict(list)
        with open(json_path) as f:
            i = 0
            for line in tqdm(f, desc=f'Loading {stage} dataset'):
                line = line.strip()
                js = BinFunc(**json.loads(line))
                self.pairs.append(js)
                bins[js.function].append(i) # add the index to bin
                i += 1
        
        self.indices = []
        self.functions = list(bins.keys())
        keys = [key for key in bins.keys() if len(bins[key]) >= 2]
        for _ in trange(max_num, desc=f'Sampling {stage} dataset'):
            key = random.choice(keys)
            x, y = random.sample(range(len(bins[key])), 2)
            y_index = bins[key][y]
            x_index = bins[key].pop(x)
            self.indices.append((x_index, y_index))
            if len(bins[key]) < 2:
                keys.remove(key)
            if len(keys) == 0:
                break
            

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x_index, y_index = self.indices[index]
        x:BinFunc = self.pairs[x_index]
        y:BinFunc = self.pairs[y_index]
        label = self.functions.index(x.function)
        return x.get_blocks(self.max_lines), y.get_blocks(self.max_lines), label