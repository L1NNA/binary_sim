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

from models.llm2vec import Qwen2BiForMNTP
from data_loaders.mntp_dataset import load_mntp_dataset

MODELS = {
    'codeqwen': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2BiForMNTP, "<|fim_middle|>"),
}

class DataCollatorForMNTP(DataCollatorForLanguageModeling):
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
    parser.add_argument('--model', required=True, help='model name',
                        choices=MODELS.keys(),
                        default='codeqwen')
    parser.add_argument("--note", type=str, default='',help="additonal note")

    # data_loader
    parser.add_argument('--data_path', type=str, default='train', help='dataset path')
    parser.add_argument("--test_data_path", type=str, default='test_o0')
    parser.add_argument("--max_seq_len", type=int, default=512,
        help="max number of tokens")

    # training configurations
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--resume", action='store_true', help="whether to resume training")
    parser.add_argument('--epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='optimizer learning rate')
    parser.add_argument('--metric', type=str, default='loss', help='best metric')

    # Parse the arguments
    args = parser.parse_args()
    
    ########## Load Data ##########
    model_path, model_cls, mask_token = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    
    ########## Load Data ##########
    if tokenizer.mask_token is None:
        tokenizer.mask_token = mask_token
    data_collator = DataCollatorForMNTP(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )
    dataset = load_mntp_dataset(tokenizer, args.max_seq_len, args.data_path)
    val_dataset = load_mntp_dataset(tokenizer, args.max_seq_len, args.test_data_path)

    ########## Training ##########
    # Define the training arguments
    output_dir = f'./model_checkpoints/mntp_{args.model}'
    warmup_steps = int(1000/args.batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        learning_rate=args.lr,
        logging_dir=join(output_dir, 'logs'),
        logging_steps=250,
        eval_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model=args.metric,
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
