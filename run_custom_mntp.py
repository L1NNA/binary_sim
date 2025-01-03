import os
import argparse
from os.path import join
from typing import Optional, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from models.llm2vec_custom import CustomQwen2BiForMNTP
from models.llm2vec import Qwen2BiForMNTP
from data_loaders.mntp_dataset import load_mntp_dataset, load_mntp_pair_dataset
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch
import re
from torch.utils.data import random_split

MODELS = {
    'codeqwen': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2BiForMNTP, "<|fim_middle|>"),
    'codeqwen_uptrained': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2BiForMNTP, "<|fim_middle|>"),
}

# python run_custom_mntp.py --model='codeqwen' --epochs=2 --use_pair --note='pair_arch_blkmask' --attention='eager' --use_arch --mask_block_token
# python run_custom_mntp.py --model='codeqwen' --epochs=2 --use_pair --note='pair_maskblocktoken' --attention='eager' --mask_block_token
# python run_custom_mntp.py --model='codeqwen' --epochs=2 --use_pair --note='pair_maskblocktoken' --attention='eager' --mask_block_token
class DataCollatorForMNTP(DataCollatorForLanguageModeling):

    def __init__(
        self,
        tokenizer: Any,
        mlm_probability: float = 0.15,
        use_pair=None,
        use_arch = None,
        mask_block_token=None,
    ):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.use_arch = use_arch
        self.use_pair = use_pair
        self.mask_block_token = mask_block_token

    

    def torch_call(self, examples) -> Dict[str, Any]:
            if self.mask_block_token:
                blk_mask = [i.pop('blk_mask') for i in examples]
                blk_mask = torch.tensor([ [0]*(len(j) - len(blk_mask[i]))+blk_mask[i] for i,j in enumerate(batch['input_ids'])]).bool()
            # Handle dict or lists with proper padding and conversion to tensor.
            if isinstance(examples[0], Mapping):
                batch = pad_without_fast_tokenizer_warning(
                    self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
                )
            else:
                batch = {
                    "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
                }

            

            # If special token mask has been preprocessed, pop it from the dict.
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            if self.mlm:
                batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
            else:
                labels = batch["input_ids"].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels

            if self.mask_block_token:
                batch['blk_mask'] = (batch['input_ids'] == self.tokenizer.convert_tokens_to_ids('<BLK>')).long()
                batch['labels'] = batch['labels'].masked_fill(blk_mask, tokenizer.convert_tokens_to_ids('<BLK>'))
                batch['input_ids'] = batch['input_ids'].masked_fill(blk_mask, self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token))

            ## generate arch embedding (similar to language embedding) for each token
            if self.use_arch: 
                seq_len = batch['input_ids'].size(-1)
                arch_pattern_x = r'( arm | mips | powerpc | x86 )'
                arch_pattern_y = r'( arm| mips| powerpc| x86)'
                map = {'arm': 0, 'mips': 1, 'powerpc': 2, 'x86': 3}
                if self.use_pair is None:
                    last_block = [self.tokenizer.decode(ex['input_ids'][-10:]) for ex in examples]
                    pattern = [self.find_pattern(arch_pattern_y, i)[1:] for i in last_block]
                    batch['arch'] = torch.tensor([map[i] for i in pattern]).reshape(-1, 1).expand(-1, batch['input_ids'].size(-1))
                else:
                    last_block_x = [self.tokenizer.decode(ex['input_ids'][:-10]) for ex in examples]
                    last_block_y = [self.tokenizer.decode(ex['input_ids'][-10:]) for ex in examples]
                    pattern_x = [self.find_pattern(arch_pattern_x, i)[1:-1] for i in last_block_x]
                    pattern_y = [self.find_pattern(arch_pattern_y, i)[1:] for i in last_block_y]
                    idx_x = [j.split(' ').index(pattern_x[i]) for i,j in enumerate(last_block_x)]
                    temp = [' '.join(j.split(' ')[idx_x[i]+1:]) + last_block_y[i] for i,j in enumerate(last_block_x)]
                    temp_tok = self.tokenizer(temp)['input_ids']
                    len_x = [seq_len - len(i) for i in temp_tok]
                    len_y = [len(i) for i in temp_tok]
                    arch_x = [torch.full((1, len_x[i]), map[pattern_x[i]]) for i in range(len(pattern_x))]
                    arch_y = [torch.full((1, len_y[i]), map[pattern_y[i]]) for i in range(len(pattern_y))]
                    batch['arch'] = torch.cat([torch.cat((arch_x[i], arch_y[i]), dim=1) for i in range(batch['input_ids'].size(0))])

            return batch
    
    def find_pattern(self, p, string):
        pattern = re.compile(p)
        pattern = pattern.search(string)
        if pattern is not None:
            return pattern.group()
        else:
            return ''

    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK, 0% random, 0% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels
    



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="")
    
    # hyperparameters
    parser.add_argument('--mlm_probability', help='probability to mask tokens',
                        type=float, default=0.15)

    # loading model
    parser.add_argument('--model', help='model name',
                        choices=MODELS.keys(),
                        default='codeqwen_uptrained')
    parser.add_argument("--note", type=str, default='',help="additonal note")
    parser.add_argument("--attention", type=str, help="attention type, leave empty for default sdpa or eager for custom")

    # data_loader
    parser.add_argument('--data_path', type=str, default='train', help='dataset path')
    parser.add_argument("--test_data_path", type=str, default='test_o0')
    parser.add_argument("--max_seq_len", type=int, default=512,
        help="max number of tokens")
    parser.add_argument("--local_model_path", type=str,)

    # training configurations
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--resume", action='store_true', help="whether to resume training")
    parser.add_argument('--epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='optimizer learning rate')
    parser.add_argument('--metric', type=str, default='loss', help='best metric')
    parser.add_argument('--use_pair', action='store_true', help='use pair dataset')
    parser.add_argument('--mask_block_token', action='store_true', help='use blk mask')
    parser.add_argument('--use_arch', action='store_true', help='use arch embedding')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    # Parse the arguments
    args = parser.parse_args()
    
    ########## Load Data ##########
    model_path, model_cls, mask_token = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left',
        truncation_side='left',
    )

    special_tokens = ['<addr>', '<byte>', '<str>', '<BLK>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.eos_token = '<EOS>'
    tokenizer.sep_token = '<SEP>'
    tokenizer.pad_token = '<PAD>'
    tokenizer.unk_token = '<unk>'

    config = AutoConfig.from_pretrained(model_path)
    if model_cls is CustomQwen2BiForMNTP:
        config._attn_implementation = 'eager'
    config.attention_dropout = args.dropout
    model = model_cls.from_pretrained(
        model_path if args.local_model_path is None else args.local_model_path,
        torch_dtype=torch.bfloat16,
        config=config,
    )

    
    ########## Load Data ##########
    if tokenizer.mask_token is None:
        tokenizer.mask_token = mask_token
    data_collator = DataCollatorForMNTP(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        use_pair=args.use_pair,
        mask_block_token=args.mask_block_token,
        use_arch=args.use_arch,
    )
    # data_collator = lambda x: DataCollatorForMNTP1(x, tokenizer)

    if args.use_pair:
        dataset = load_mntp_pair_dataset(tokenizer, args.max_seq_len, stage=args.data_path, max_num=200_000)
        total_length = len(dataset)
        train_length = int(total_length * 0.9)
        val_length = total_length - train_length
        torch.manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
        # val_dataset = load_mntp_pair_dataset(tokenizer, args.max_seq_len, stage=args.test_data_path, max_num=10)
    else:
        dataset = load_mntp_dataset(tokenizer, args.max_seq_len, stage=args.data_path)
        total_length = len(dataset)
        train_length = int(total_length * 0.9) if args.data_path == 'train' else int(total_length * 0.3)
        arb_length = 0 if args.data_path == 'train' else int(total_length*0.6)
        val_length = total_length - train_length - arb_length
        torch.manual_seed(42)
        train_dataset, val_dataset, _ = random_split(dataset, [train_length, val_length, arb_length])
        # val_dataset = load_mntp_dataset(tokenizer, args.max_seq_len, stage=args.test_data_path)
    
    

    ########## Training ##########
    # Define the training arguments
    output_dir = f'./model_checkpoints/custom_mntp_{args.model}'
    if args.note:
        output_dir += '_' + args.note
    warmup_steps = 500
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type = "cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 2},
        learning_rate=args.lr,
        logging_dir=join(output_dir, 'logs'),
        logging_steps=250,
        eval_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model=args.metric,
        remove_unused_columns=True,
        # gradient_accumulation_steps=4,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)
