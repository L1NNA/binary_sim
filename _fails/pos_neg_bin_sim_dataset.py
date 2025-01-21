from torch.utils.data import Dataset
from tqdm import tqdm, trange
import json
import torch
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import random
import re
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
    
    
def pairwise_collate(batch, tokenizer, max_blocks, max_length=512, require_blk_mask = False, arch_embed=False, padding = True):
    x_batches, y_batches, labels = zip(*batch)
    blk_token = tokenizer.convert_tokens_to_ids('<BLK>')

    def find_pattern(p, string):
        pattern = re.compile(p)
        pattern = pattern.search(string)
        if pattern is not None:
            return pattern.group()
        else:
            return ''
    
    x_inputs = get_tokens(x_batches, tokenizer, max_blocks, max_length, padding = padding, get_blk_mask=require_blk_mask, get_arch=arch_embed)
    y_inputs = get_tokens(y_batches, tokenizer, max_blocks, max_length, padding = padding, get_blk_mask=require_blk_mask, get_arch=arch_embed)

    result = {
            'input_ids': x_inputs['input_ids'],
            'attention_mask': x_inputs['attention_mask'],
            'y_input_ids': y_inputs['input_ids'],
            'y_attention_mask': y_inputs['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    
    if require_blk_mask:
        result['blk_mask'] = x_inputs['blk_mask']
        result['y_blk_mask'] = y_inputs['blk_mask']
        # result['blk_mask'] = (x_inputs['input_ids'] == blk_token).long()
        # result['y_blk_mask'] = (y_inputs['input_ids'] == blk_token).long()
        
    
    if arch_embed:
        result['arch'] = x_inputs['arch']
        result['y_arch'] = y_inputs['arch']
        # arch_pattern = r'( arm| mips| powerpc| x86)'
        # map = {'arm': 0, 'mips': 1, 'powerpc': 2, 'x86': 3}
        # last_blocks = [' '.join(i.split()[-5:]) for i in x_batches]
        # pattern = [find_pattern(arch_pattern, i)[1:] for i in last_blocks]
        # result['arch'] = torch.tensor([map[i] for i in pattern]).reshape(-1, 1).expand(-1, result['input_ids'].size(-1)).clone()

        # last_blocks = [' '.join(i.split()[-5:]) for i in y_batches]
        # pattern = [find_pattern(arch_pattern, i)[1:] for i in last_blocks]
        # result['y_arch'] = torch.tensor([map[i] for i in pattern]).reshape(-1, 1).expand(-1, result['y_input_ids'].size(-1)).clone()
    # print(result['blk_mask'].shape)
    return result