import json
import argparse
from datetime import datetime
from os.path import join
import os
import random
from typing import Dict, Union, Tuple, Any

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import random_split

from models.codet5p_models import CodeT5PEncoderForSequenceEmbedding, CodeT5PForSequenceEmbedding
from models.bert_models import GraphCodeBERTForSequenceEmbedding
from models.qwen_models import Qwen2ForSequenceEmbedding, preload_qwen2_from_causal_lm, CustomQwen2ForSequenceEmbedding
from models.llm2vec import Qwen2MNTPForSequenceEmbedding
from data_loaders.pos_neg_bin_sim_dataset import pairwise_collate
from data_loaders.sim_cse import SimCSEDataset
from layers.loss import SimeCSELoss

from run_test_retrieval import get_embeddings
from torch.optim import AdamW

## # python run_simcse.py --model='codeqwen2vec' --local_model_path='model_checkpoints/mntp_codeqwen/checkpoint-55804' --epochs=2 --max_length=1024
## # python run_simcse.py --model='codeqwen' --epochs=2 --max_length=1024

# python run_simcse.py --model='customcodeqwen2vec' --local_model_path='model_checkpoints/mntp_codeqwen/checkpoint-55804' --epochs=2 --arch_embed --require_blk_mask --note='arch_blkmask' --attention='eager' 
# python .\run_simcse.py --model='customcodeqwen2vec' --local_model_path='model_checkpoints/mntp_codeqwen/checkpoint-55804' --epochs=2 --dropout=0.1 --unsupervised --attention='eager' --note='dropout_unsupervised'

models = {
    'qwen_emb': ('Alibaba-NLP/gte-Qwen2-1.5B-instruct', Qwen2ForSequenceEmbedding, None),
    'codet5p-110m-embedding': ('Salesforce/codet5p-110m-embedding', CodeT5PEncoderForSequenceEmbedding, None),
    'codeqwen': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding, preload_qwen2_from_causal_lm),
    'codeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', Qwen2MNTPForSequenceEmbedding, preload_qwen2_from_causal_lm),
    'graphcodebert': ('microsoft/graphcodebert-base', GraphCodeBERTForSequenceEmbedding, None),
    'uptrainedcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2ForSequenceEmbedding, None),
    'customcodeqwen2vec': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', CustomQwen2ForSequenceEmbedding, preload_qwen2_from_causal_lm), 
}

class SimCSETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = SimeCSELoss()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # TODO support attention dropout
        output = model(inputs['input_ids'], inputs['attention_mask'])
        y_embs = model(inputs['y_input_ids'], inputs['y_attention_mask']).embedding
        output.y_embedding = y_embs

        loss = self.loss_function(output.embedding, y_embs)

        if return_outputs:
            return loss, output

        return loss


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # === model ===
    parser.add_argument("--model", choices=models.keys(),
                        default='qwen_emb', help="The model name")
    parser.add_argument("--local_model_path", type=str,)
    parser.add_argument("--local_tokenizer_path", type=str)
    parser.add_argument("--note", type=str)

    # data_loader
    parser.add_argument(
        "--data", type=str, default='train'
    )
    parser.add_argument(
        "--max_blocks", type=int, default=0,
        help="max lines"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="max number of tokens per line"
    )
    parser.add_argument(
        "--require_blk_mask", action='store_true',
        help="whether to use block mask for <BLK> token pooling"
    )
    parser.add_argument(
        "--arch_embed", action='store_true',
        help="whether to use architecture embedding"
    )

    # training configurations
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--resume", action='store_true', help="whether to resume training")
    parser.add_argument('--epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument("--max_pairs", type=int, default=75_000)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument("--attention", type=str, help="attention type, leave empty for default sdpa or eager for custom")
    parser.add_argument("--unsupervised", action="store_true", help="use unsupervised x_emb training from dropout")
    parser.add_argument("--use_flex", action="store_true", help="use flex attention")
    
    # Parse the arguments
    args = parser.parse_args()

    
    # ================= Load Model ======================
    model_path, model_cls, pre_load = models[args.model]
    config = AutoConfig.from_pretrained(model_path)
    if args.attention:
        config._attn_implementation = args.attention
    config.attention_dropout = args.dropout
    if args.unsupervised:
        config.use_unsupervised = True
    if args.use_flex:
        config.use_flex = True
    if args.arch_embed:
        config.arch_embed = True
    if args.require_blk_mask:
        config.require_blk_mask = True
    if pre_load:
        model = pre_load(model_path, model_cls, args.local_model_path, config)
    else:
        model = model_cls.from_pretrained(
            model_path if args.local_model_path is None else args.local_model_path,
            trust_remote_code=True,
            torch_dtype = torch.bfloat16,
            config=config
        ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if args.local_tokenizer_path is None else args.local_tokenizer_path,
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
    
    padding = 'max_length' if args.use_flex else True

    ########## Load Data ##########
    torch.manual_seed(42)
    random.seed(42)
    dataset = SimCSEDataset(args.data, args.max_blocks, args.max_pairs)
    total_length = len(dataset)
    train_length = int(total_length * 0.9)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

    ########## Training ##########
    # parameter groups
    adjust_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "arch_proj" in name or "embed_arch" in name or "blk_proj" in name:  # Check if the parameter belongs to `q_proj`
            adjust_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': adjust_params, 'lr': 1e-4},
        {'params': other_params, 'lr': 1e-5},
    ]
    custom_optimizer = AdamW(param_groups, eps=1e-7)

    # Define the training arguments
    output_dir = f'./model_checkpoints/simcse_{args.model}'
    if args.note:
        output_dir += '_' + args.note
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
    collator_wrapper = lambda x:pairwise_collate(x, tokenizer, args.max_blocks, args.max_length, require_blk_mask=args.require_blk_mask, arch_embed = args.arch_embed, padding=padding)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-4)
    trainer = SimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator_wrapper,
        callbacks=[early_stopping_callback],
        optimizers = (custom_optimizer, None)
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)

    ########## Inference ##########
    print(get_embeddings(
        model, tokenizer, './datasets', 'o0', 'o1', 1_000, args.max_length, args.max_blocks, args.test_batch_size
    ))

