import json
import random
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm import tqdm, trange

from utils import BinFunc, get_tokens
import torch
import os


class SimCSEDataset(Dataset):

    def __init__(self, stage:str='train', max_blocks=0, max_num=10000):
        self.max_blocks = max_blocks
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
        return x.get_blocks(self.max_blocks), y.get_blocks(self.max_blocks), label

def SimCSE_collate(batch, tokenizer, max_blocks, max_length=512, require_blk_mask = False, arch_embed=False, padding = True, prompt=False):
    x_batches, y_batches, labels = zip(*batch)
    
    if prompt:
        if '<BLK>' in x_batches[0] or '<blk>' in x_batches[0]:
            x_batches = [i+f'\n The above assembly function is compiled with the setting: {i.split(" <BLK> ")[-1]}, provide its semantics' for i in x_batches]
            y_batches = [i+f'\n The above assembly function is compiled with the setting: {i.split(" <BLK> ")[-1]}, provide its semantics' for i in y_batches]
        else:
            x_batches = [i+f'\n The above assembly function is compiled from optimization level {i.split()[-1]}, provide its semantics' for i in x_batches]
            y_batches = [i+f'\n The above assembly function is compiled from optimization level {i.split()[-1]}, provide its semantics' for i in y_batches]
    
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
        
    
    if arch_embed:
        result['arch'] = x_inputs['arch']
        result['y_arch'] = y_inputs['arch']
    return result
    

class SimCSEDatasetUnsupervised(Dataset):

    def __init__(self, stage:str='train', max_blocks=0, max_num=10000):
        self.max_blocks = max_blocks
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        x:BinFunc = self.pairs[index]
        return x.get_blocks(self.max_blocks), 1
    
def simcse_unsupervised_collate(examples, tokenizer, max_length=1024, require_blk_mask=True):
    input, label = zip(*examples)
    x = tokenizer(input, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    if require_blk_mask:
        blk_mask = (x['input_ids'] == tokenizer.convert_tokens_to_ids('<BLK>')).long()
    return {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'blk_mask': blk_mask, 'labels': torch.tensor(label, dtype=torch.long)}
    
class SimCSEDatasetUnsupervisedPair(Dataset):

    def __init__(self, stage:str='train', max_line = 0, max_num=50000):
        self.bins = defaultdict(list)
        self.max_line = max_line
        self.pairs = []
        path = os.path.join('datasets', f'{stage}.jsonl')
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
            y += [' <EOS>']
            return (x, y)
        
def simcse_unsupervised_pair_collate(batch, tokenizer, max_blocks=0, max_length=128):
    ### collate function for paired causal, where <SEP> token is inserted in between
    ### first tokenize source and target, then check max_length for each. if both are over, concat them. if both are under, 
    source, target = zip(*batch)
    b = len(batch)
    blk_token = tokenizer.convert_tokens_to_ids('<BLK>')

    inputs_source = get_tokens(source, tokenizer, max_blocks, int(max_length//2))
    inputs_target = get_tokens(target, tokenizer, max_blocks, int(max_length//2))

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
    labels = torch.arange(b, device=input_ids.device)

    blk_mask = (input_ids == blk_token).long()


    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'blk_mask': blk_mask
    }
    return result