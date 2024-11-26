import json
import gzip
import os
import random
import pickle
from collections import defaultdict
from tqdm import trange, tqdm
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_tokens, BinFunc


class RetrievalDataset(Dataset):
    
    def __init__(self, path:str, max_line = 0,keys:list=None):
        self.bins = defaultdict(list)
        with open(path) as f:
            for line in f:
                # js = json.loads(line)
                # self.bins[js['function']].append(js['blocks'])
                js = BinFunc(**json.loads(line))
                self.bins['blocks'].append(js.get_blocks(max_blocks = max_line))
                self.bins['function'].append(js.function)
                self.bins['file'].append(js.file)
        self.keys = keys

    def __len__(self):
        if self.keys is None:
            return len(self.bins)
        return len(self.keys)
    
    def __getitem__(self, index):
        if self.keys is None:
            blocks = self.bins['blocks'][index]
            return blocks
        func = self.keys[index]
        blocks = self.bins[func][0]
        return blocks, func
    
    def get_all(self, keys=None, join_blocks=False):
        result = []
        if keys is None:
            for func in self.bins['blocks']:
                if join_blocks:
                    blocks = ' '.join(func)
                    result.append(blocks)
                    continue
                result.append(' '.join(func.split())+' <EOS>')
        else:
            for func in keys:
                idx = self.bins['function'].index(func)
                blocks = self.bins['blocks'][idx]
                if join_blocks:
                    blocks = ' '.join(blocks)
                result.append(blocks)
        return result
    

        
def get_instructions(batches, tokenizer, max_blocks, max_length):
    ins = []
    '''
        batches: [batch, blocks, lines]
    '''
    for blocks in batches:
        if len(blocks) > max_blocks:
            # select middle
            remant = (len(blocks) - max_blocks) // 2
            blocks = blocks[remant:remant+max_blocks]
        elif len(blocks) < max_blocks:
            for _ in range(max_blocks - len(blocks)):
                blocks.append("")
        ins.extend(blocks)  

    inputs = tokenizer(
        ins, return_tensors='pt',
        padding=True, truncation=True, max_length=max_length
    )
    return inputs
    
    
def line_collate(batch, tokenizer, max_blocks=64, max_length=128):
    s_batches, _, _ = zip(*batch)
    b = len(s_batches)
    
    ins = get_instructions(s_batches, tokenizer, max_blocks, max_length)

    result = {
        'input_ids': ins['input_ids'].reshape(b, max_blocks, -1),
        'attention_mask': ins['attention_mask'].reshape(b, max_blocks, -1)
    }
    return result
