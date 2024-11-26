import json
import argparse
from datetime import datetime
from os.path import join
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import TrainingArguments, Trainer
from torch.utils.data import random_split


from models.codet5p_models import CodeT5PEmbeddingModel, CodeT5PModel
from models.bert_models import GraphCodeBERTEmbedding
from data_loaders.pos_neg_bin_sim_dataset import pairwise_collate
from data_loaders.sim_cse import SimCSEDataset
from models.qwen_models import Qwen2Model

from run_test_retrieval import get_embeddings


models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2Model),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEmbeddingModel),
    'codet5p-220m': ("Salesforce/codet5p-220m", CodeT5PModel),
    'codet5p-770m': ("Salesforce/codet5p-770m", CodeT5PModel),
    'codeqwen': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2Model),
    'qwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2Model),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTEmbedding),
}


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # === model ===
    parser.add_argument("--model", choices=models.keys(),
                        default='qwen_emb', help="The model name")
    parser.add_argument("--local_model_path", type=str,)
    parser.add_argument("--local_tokenizer_path", type=str)

    # data_loader
    parser.add_argument(
        "--max_blocks", type=int, default=0,
        help="max lines"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="max number of tokens per line"
    )

    # training configurations
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--resume", action='store_true', help="whether to resume training")
    parser.add_argument('--epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='optimizer learning rate')
    parser.add_argument("--max_pairs", type=int, default=50_000)
    
    # Parse the arguments
    args = parser.parse_args()

    
    # ================= Load Model ======================
    model_path, model_cls = models[args.model]
    config = AutoConfig.from_pretrained(model_path if args.local_model_path is None else args.local_model_path)
    config.torch_dtype = torch.bfloat16
    model = model_cls(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if args.local_tokenizer_path is None else args.local_tokenizer_path,
        trust_remote_code=True,
        padding_side='left'
    )

    ########## Load Data ##########
    dataset = SimCSEDataset('train', args.max_blocks, args.max_pairs)
    total_length = len(dataset)
    train_length = int(total_length * 0.9)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

    ########## Training ##########
    # Define the training arguments
    output_dir = f'./model_checkpoints/simcse_{args.model}'
    warmup_steps = int(1000/args.batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        learning_rate=args.lr,
        logging_steps=250,
        eval_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model='loss',
        local_rank=int(os.environ.get('LOCAL_RANK', -1))
    )

    # Define the Trainer
    # metric_wrapper = lambda p: compute_metrics(p, join(output_dir, 'valid.jsonl'))
    collator_wrapper = lambda x:pairwise_collate(x, tokenizer, args.max_blocks, args.max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator_wrapper,
        # compute_metrics=metric_wrapper,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)

    ########## Inference ##########
    print(get_embeddings(
        model, tokenizer, './datasets', 'o0', 'o1', 1_000, args.max_length, args.max_blocks, args.test_batch_size
    ))

