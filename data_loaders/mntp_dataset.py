import json
from tqdm import tqdm
from os.path import exists
import os

from datasets import Dataset, load_from_disk
from collections import defaultdict
from utils import BinFunc, get_tokens
import random
from tqdm import trange, tqdm
import torch


def load_mntp_dataset(tokenizer, max_length, stage='train'):
    tokenized_path = os.path.join('datasets', f'{stage}_tokenized_{max_length}')
    if exists(tokenized_path):
        tokenized_dataset = load_from_disk(tokenized_path)
    else:
        json_path = f'./datasets/{stage}.jsonl'
        functions = {"functions": []}
        with open(json_path) as f:
            for line in tqdm(f, desc=f'Loading {stage} dataset'):
                js = BinFunc(**json.loads(line))
                functions['functions'].append(js.get_blocks(0, use_blk_token=False))

        dataset = Dataset.from_dict(functions)

        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["functions"], truncation=True, max_length=max_length)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['functions'])
        tokenized_dataset.save_to_disk(tokenized_path)
    return tokenized_dataset

def load_mntp_pair_dataset(tokenizer, max_length, max_blocks = 0, stage='train', max_num=50000):
    tokenized_path = os.path.join('datasets', f'{stage}_pair_tokenized_{max_length}')
    if exists(tokenized_path):
        tokenized_dataset = load_from_disk(tokenized_path)
    else:
        json_path = f'./datasets/{stage}.jsonl'
        bins = defaultdict(list)
        pairs = []
        with open(json_path) as f:
            i = 0
            for line in tqdm(f, desc=f'Loading dataset'):
                line = line.strip()
                js = BinFunc(**json.loads(line))
                pairs.append(js)
                bins[js.function].append(i)
                i += 1

        blk_token = tokenizer.convert_tokens_to_ids('<BLK>')
        results = {'functions': []}
        keys = [key for key in bins.keys() if len(bins[key]) >= 2]
        for _ in trange(max_num, desc=f'Sampling dataset'):
            key = random.choice(keys)
            index, pos_index = random.sample(range(len(bins[key])), 2)
            pos_index = bins[key][pos_index]
            index = bins[key].pop(index)
            
            ### first get the pair, then tokenize them using gettokens(), then concat them based on max_length
            source = pairs[index]
            target = pairs[pos_index]
            results['functions'].append((source.get_blocks(max_blocks=max_blocks), target.get_blocks(max_blocks=max_blocks)))

            if len(bins[key]) < 2:
                keys.remove(key)
            if len(keys) == 0:
                break


        dataset = Dataset.from_dict(results)

        # Tokenize the dataset
        def get_tokenized_concat_pairs(examples):
            blk_token = tokenizer.convert_tokens_to_ids('<BLK>')

            examples = examples['functions']
            source, target = zip(*examples)
            b = len(examples)
            input_ids =[]
            attention_mask = []
            blk_mask = []

            inputs_source = tokenizer(source, truncation=True, max_length=int(max_length//2))
            inputs_target = tokenizer(target, truncation=True, max_length=int(max_length//2))
            seq_source, mask_source = inputs_source['input_ids'], inputs_source['attention_mask']
            seq_target, mask_target = inputs_target['input_ids'], inputs_target['attention_mask']

            for i in range(b):
                
                combined_input_ids = seq_source[i] + seq_target[i]
                combined_attention_mask = mask_source[i] + mask_target[i]
                input_ids.append(combined_input_ids)
                attention_mask.append(combined_attention_mask)
                blk_mask.append([1 if i==blk_token else 0 for i in combined_input_ids])            

            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'blk_mask': blk_mask
            }
            return result

        tokenized_dataset = dataset.map(get_tokenized_concat_pairs, batched=True, remove_columns=['functions'])
        tokenized_dataset.save_to_disk(tokenized_path)
    return tokenized_dataset


