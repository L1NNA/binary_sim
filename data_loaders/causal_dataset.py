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
from transformers import DataCollatorForLanguageModeling
from utils import get_tokens, BinFunc


class CausalDatasetPair(Dataset):
    '''
    We sample a pair of positive pairs for translation purpose
    pair of cross opt, cross compiler, cross arch
    each sample is a concat of two functions, with a <SEP> token in between
    eventually put an EOS token at the end of the sequence

    bi-directional inverse ALiBi?
    
    
    '''
    
    def __init__(self, path:str, max_line = 0, max_num=50000):
        self.bins = defaultdict(list)
        self.max_line = max_line
        self.pairs = []
        with open(path) as f:
            i = 0
            for line in tqdm(f, desc=f'Loading dataset'):
                line = line.strip()
                js = BinFunc(**json.loads(line))
                self.pairs.append(js)
                self.bins[js.function].append(i)
                i += 1
                
        self.indices = []
        keys = [key for key in self.bins.keys() if len(self.bins[key]) >= 2]
        for _ in trange(max_num, desc=f'Sampling dataset'):
            key = random.choice(keys)
            index, pos_index = random.sample(range(len(self.bins[key])), 2)
            pos_index = self.bins[key][pos_index]
            index = self.bins[key].pop(index)
            self.indices.append((index, pos_index, 1))
            if len(self.bins[key]) < 2:
                keys.remove(key)
            if len(keys) == 0:
                break

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        x_index, y_index, label = self.indices[index]
        x_js = self.pairs[x_index]
        y_js = self.pairs[y_index]
        x = x_js.get_blocks(self.max_line)
        y = y_js.get_blocks(self.max_line, remove_last = True)
        last_block_y = y_js.blocks[-1]
        if self.max_line == 0:
            x += ' <SEP> '+last_block_y
            y += ' <EOS>'
            # x = ' '.join(x)
            # y = ' '.join(y)
            return (x, y)
        else:
            x += [' <SEP> ']
            y += [' <EOS> ']
            return (x, y)
    
    def get_all(self, join_blocks = True, max_blocks = 50):
        result = []
        for func in self.pairs:
            if join_blocks:
                result.append(func.get_blocks(max_blocks = 0))
            else:
                result.append(func.get_blocks(max_blocks = max_blocks))
        return result


class CausalDataset(Dataset):
    
    def __init__(self, path:str, max_line = 0):
        self.bins = defaultdict(list)
        with open(path) as f:
            for line in f:
                js = BinFunc(**json.loads(line))
                self.bins['blocks'].append(js.get_blocks(max_blocks = max_line))
                self.bins['function'].append(js.function)
                self.bins['file'].append(js.file)

    def __len__(self):
        return len(self.bins['file'])
    
    def __getitem__(self, index):
        blocks = self.bins['blocks'][index]
        # func = self.bins['function'][index]
        # file = self.bins['file'][index]
        return blocks
    
    def get_all(self, join_blocks = True):
        result = []
        for func in self.bins['blocks']:
            if join_blocks:
                blocks = ' '.join(func)
                result.append(blocks)
                continue
            result.append(' '.join(func.split())+' <EOS>')
        return result

def tokenize(element, tokenizer, max_length):
    outputs = tokenizer(
        element[0],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}
    
    
def causal_collate(batch, tokenizer, max_blocks=0, max_length=128):
    b = len(batch)
    
    inputs = get_tokens(batch, tokenizer, max_blocks, max_length)
    labels = inputs['input_ids'].clone()
    
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    result = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels
    }
    return result

def causal_pair_collate(batch, tokenizer, max_blocks=0, max_length=128):
    ### collate function for paired causal, where <SEP> token is inserted in between
    ### first tokenize source and target, then check max_length for each. if both are over, concat them. if both are under, 
    source, target = zip(*batch)
    b = len(batch)
    
    inputs_source = get_tokens(source, tokenizer, max_blocks, int(max_length/2))
    inputs_target = get_tokens(target, tokenizer, max_blocks, int(max_length/2))
    
    seq_source, mask_source = inputs_source['input_ids'], inputs_source['attention_mask']
    seq_target, mask_target = inputs_target['input_ids'], inputs_target['attention_mask']
    
    input_ids =[]
    attention_mask=[]
    
    for i in range(b):
        # if not (s['attention_mask'][i] == 0).any().item() and not (t['attention_mask'][i] == 0).any().item():
        #     print(True)
        #     input_ids.append(torch.cat((s['input_ids'][i], t['input_ids'][i])))
        #     attention_mask.append(torch.cat((s['attention_mask'][i], t['attention_mask'][i])))
        # else:
            # print(False)
        non_zero_input_1 = seq_source[i][mask_source[i].bool()]
        non_zero_input_2 = seq_target[i][mask_target[i].bool()]

        combined_input_ids = torch.cat((non_zero_input_1, non_zero_input_2))
        combined_attention_mask = torch.cat(
            (torch.ones(non_zero_input_1.size(0)), torch.ones(non_zero_input_2.size(0))), dim=0
        )

        # Padding to match the original length of input sequences
        total_length = seq_source[i].size(0) + seq_target[i].size(0)
        padding_length = total_length - combined_input_ids.size(0)

        combined_input_ids = torch.cat((torch.ones(padding_length, dtype=seq_source.dtype)*tokenizer.pad_token_id, combined_input_ids))
        combined_attention_mask = torch.cat((torch.zeros(padding_length, dtype=mask_source.dtype)*tokenizer.pad_token_id, combined_attention_mask))

        input_ids.append(combined_input_ids)
        attention_mask.append(combined_attention_mask)

    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = input_ids.clone()
    
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    return result