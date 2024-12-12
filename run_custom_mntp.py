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
from torch.nn.parallel import DistributedDataParallel as DDP

MODELS = {
    'codeqwen': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2BiForMNTP, "<|fim_middle|>"),
    'trained_codeqwen': (os.path.join('model_checkpoints', 'causal_codeqwen', 'checkpoint-8672'), CustomQwen2BiForMNTP, "<|fim_middle|>"),
}

class DataCollatorForMNTP(DataCollatorForLanguageModeling):
    def torch_call(self, examples) -> Dict[str, Any]:
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
            batch['blk_mask'] = (batch['input_ids'] == self.tokenizer.convert_tokens_to_ids('<BLK>')).long()
            return batch

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
                        default='codeqwen')
    parser.add_argument("--note", type=str, default='',help="additonal note")

    # data_loader
    parser.add_argument('--data_path', type=str, default='train', help='dataset path')
    parser.add_argument("--test_data_path", type=str, default='test_o0')
    parser.add_argument("--max_seq_len", type=int, default=1024,
        help="max number of tokens")

    # training configurations
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--resume", action='store_true', help="whether to resume training")
    parser.add_argument('--epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='optimizer learning rate')
    parser.add_argument('--metric', type=str, default='loss', help='best metric')

    # Parse the arguments
    args = parser.parse_args()
    
    ########## Load Data ##########
    model_path, model_cls, mask_token = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-Coder-0.5B-Instruct',
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

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    # model = nn.DataParallel(model)
    
    
    ########## Load Data ##########
    if tokenizer.mask_token is None:
        tokenizer.mask_token = mask_token
    data_collator = DataCollatorForMNTP(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )
    # data_collator = lambda x: DataCollatorForMNTP1(x, tokenizer)

    dataset = load_mntp_pair_dataset(tokenizer, args.max_seq_len, stage=args.data_path, max_num=200000)
    val_dataset = load_mntp_pair_dataset(tokenizer, args.max_seq_len, stage=args.test_data_path, max_num=200000)
    if 'blk_mask' in dataset[0].keys():
        dataset = dataset.remove_columns(["blk_mask"])
        val_dataset = val_dataset.remove_columns(["blk_mask"])

    
    
    # val_dataset = load_mntp_dataset(tokenizer, args.max_seq_len, stage=args.test_data_path)
    # dataset = load_mntp_dataset(tokenizer, args.max_seq_len, stage=args.data_path)
    
    

    ########## Training ##########
    # Define the training arguments
    output_dir = f'./model_checkpoints/custom_mntp_{args.model}'
    warmup_steps = int(1000/args.batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        # lr_scheduler_type = "cosine_with_restarts",
        # lr_scheduler_kwargs={"num_cycles": 2},
        learning_rate=args.lr,
        logging_dir=join(output_dir, 'logs'),
        logging_steps=250,
        eval_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model=args.metric,
        # remove_unused_columns=True,
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
