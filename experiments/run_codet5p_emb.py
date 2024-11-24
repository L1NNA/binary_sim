import json
import argparse
from datetime import datetime
from os.path import join
import numpy as np
import os
import random

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn import DataParallel
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, recall_score, precision_score

from models.coe_cos_sim import CoECosSim
from models.codet5p_embedding import CodeT5PEmbeddingModel
from data_loaders.pos_neg_bin_sim_dataset import BinSimDataset, pairwise_collate
from data_loaders.pos_bin_sim_dataset import PosBinSimDataset
from data_loaders.test_retrieval_dataset import RetrievalDataset

from utils import compute_metrics
from utils import find_common_elements, calculate_mrr, compute_retrieval_metrics, get_tokens
from evaluation.retrieval_utils import get_embeddings


models = {
    'qwen_emb': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'codet5p-110m-embedding': 'Salesforce/codet5p-110m-embedding',
    'codet5p-220m': "Salesforce/codet5p-220m",
    'codet5p-770m': "Salesforce/codet5p-770m",
}


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="The batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32,
        help="The batch size."
    )
    parser.add_argument(
        "--model", choices=models.keys(), default='codet5p-110m-embedding',
        help="The model name"
    )
    parser.add_argument(
        "--max_lines", type=int, default=0,
        help="max lines"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="max number of tokens per line"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=50000,
        help="max number of tokens per line"
    )
    parser.add_argument(
        "--resume", action='store_true',
        help="resume"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="number of epochs"
    )
    parser.add_argument(
        "--source_dataset", type=str, default='obf_none',
        help="source dataset for retrieval"
    )
    parser.add_argument(
        "--target_dataset", type=str, default='obf_all',
        help="target dataset for retrieval"
    )
    parser.add_argument(
        "--pool_size", type=int, default=1000,
        help="pool size for retrieval"
    )
    # Parse the arguments
    args = parser.parse_args()

    
    ################# Load PLM ##############################
    checkpoint = models[args.model]

    if args.model == 'qwen_emb':
        backbone = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        backbone = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    if args.model == 'codet5p-220m' or args.model == 'codet5p-770m':
        vocab = backbone.config.vocab_size
        dim =  backbone.config.hidden_size
        backbone.encoder.embed_tokens = nn.Embedding(vocab, dim)
        backbone.decoder.embed_tokens = nn.Embedding(vocab, dim)
        with torch.no_grad():
            backbone.encoder.embed_tokens.weight.copy_(backbone.shared.weight)
            backbone.decoder.embed_tokens.weight.copy_(backbone.shared.weight)
    elif args.model == 'codet5p-110m-embedding':
        del backbone.shared

    ########## Load Data ##########
    dataset = PosBinSimDataset('train', args.max_lines, args.max_pairs)
    total_length = len(dataset)
    train_length = int(total_length * 0.9)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    
    model = CodeT5PEmbeddingModel(backbone)

    ########## Training ##########
    # Define the training arguments
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f'./model_checkpoints/infonce_{args.model}'
    warmup_steps = int(1000/args.batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        learning_rate=0.000001,
        logging_dir=join(output_dir, 'logs', current_time),
        logging_steps=250,
        # eval_steps=1000,
        # eval_strategy="steps",
        eval_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        # save_steps=500,
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model='loss'
    )

    # Define the Trainer
    # metric_wrapper = lambda p: compute_metrics(p, join(output_dir, 'valid.jsonl'))
    collator_wrapper = lambda x:pairwise_collate(x, tokenizer, args.max_lines, args.max_length)
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
    path = './datasets'
    source_path = join(path, 'test_'+args.source_dataset+'.jsonl')
    target_path = join(path, 'test_'+args.target_dataset+'.jsonl')
    source_dataset = RetrievalDataset(source_path, keys=None)
    target_dataset = RetrievalDataset(target_path, keys=None)

    ## sample pool_size number of functions that exist in both source and target
    source_keys = list(source_dataset.bins.keys())
    target_keys = list(target_dataset.bins.keys())
    common_keys = find_common_elements([source_keys, target_keys])
    assert len(common_keys) >= args.pool_size, "pool size too large, not enough functions with this setup"
    sampled_keys = random.sample(common_keys, args.pool_size) 

    source_funcs = source_dataset.get_all(sampled_keys, args.max_lines==0)
    target_funcs = target_dataset.get_all(sampled_keys, args.max_lines==0)
    
    query_embs, value_embs = get_embeddings(
        source_funcs, target_funcs,
        model, 
        tokenizer, args.max_length, 
        test_batch_size=args.test_batch_size,
        max_blocks=args.max_lines,
        task_source=None
    )
    
    scores = F.cosine_similarity(query_embs.unsqueeze(1), value_embs.unsqueeze(0), dim=2).numpy()
    relevance = np.arange(query_embs.size(0))
    print(f'{args.source_dataset}->{args.target_dataset}')
    print(compute_retrieval_metrics(scores, relevance))

