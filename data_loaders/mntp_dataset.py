import json
from tqdm import tqdm
from os.path import exists

from datasets import Dataset, load_from_disk

from utils import BinFunc


def load_mntp_dataset(tokenizer, max_length, stage='train'):
    tokenized_path = f'./datasets/{stage}_tokenized'
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
