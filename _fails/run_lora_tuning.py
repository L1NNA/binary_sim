import json
import argparse
from datetime import datetime
from os.path import join
import numpy as np

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
from models.prefix_sft import PrefixSFT, Sanity
from data_loaders.pos_neg_bin_sim_dataset import BinSimDataset, pairwise_collate
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, LoraConfig
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")


def compute_metrics(pred, output_file=None):
    preds = np.where(pred.predictions >=0, 1, 0)  # Get the predicted class by selecting the max logit
    labels = np.where(pred.label_ids == -1, 0, 1)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    metrics = {
        'accuracy': acc,
        'f1':f1,
        'recall':recall,
        'precision':precision
    }
    if output_file is not None:
        with open(output_file, 'a') as of:
            of.write(json.dumps(metrics)+'\n')
    return metrics


models = {
    'qwen_emb': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'codet5p-110m-embedding': 'Salesforce/codet5p-110m-embedding',
    'codet5p-220m': "Salesforce/codet5p-220m",
    'codet5p-770m': "Salesforce/codet5p-770m",
    'jina_emb': 'jinaai/jina-embeddings-v2-base-code'
}


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--batch_size", type=int, default=12,
        help="The batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=64,
        help="The batch size."
    )
    parser.add_argument(
        "--model", choices=models.keys(), default='jina_emb',
        help="The model name"
    )
    parser.add_argument(
        "--max_lines", type=int, default=0,
        help="max lines"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="max number of tokens per line"
    )
    # Parse the arguments
    args = parser.parse_args()

    
    ################# Load PLM ##############################
    checkpoint = models[args.model]

    if args.model == 'qwen_emb' or args.model == 'jina_emb':
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
    dataset = BinSimDataset('train', max_num=100000)
    total_length = len(dataset)
    train_length = int(total_length * 0.8)
    val_length = int(total_length * 0.1)
    test_length = total_length - train_length - val_length
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length])
    
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # model = PrefixSFT(backbone, hidden_size = backbone.embed_tokens.weight.shape[-1])


    lora_modules = []
    for n, m in backbone.named_modules():
        if type(m) is nn.modules.linear.Linear and 'mlp' in n:
            lora_modules.append(n)

    lora_config = LoraConfig(
        # target_modules=[r"model.layers.%d.mlp.gate_proj", r"model.layers.%d.mlp.up_proj", r"model.layers.%d.mlp.down_proj"]
        # target_modules=r'.*\.mlp\.%s_proj'
        target_modules = lora_modules
    )

    lora_model = get_peft_model(backbone, lora_config)
    model = Sanity(lora_model)
    lora_model.print_trainable_parameters()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f'./checkpoints/PrefixTuning_{args.model}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=join(output_dir, 'logs', current_time),
        logging_steps=250,
        # eval_steps=1000,
        # eval_strategy="steps",
        evaluation_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        # save_steps=500,
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model='accuracy'
    )

    # Define the Trainer
    metric_wrapper = lambda p: compute_metrics(p, join(output_dir, 'valid.jsonl'))
    collator_wrapper = lambda x:pairwise_collate(x, tokenizer, args.max_lines, args.max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator_wrapper,
        compute_metrics=metric_wrapper,
    )
    
    
    # ckp = os.path.join(os.getcwd(), 'checkpoints', 'PrefixTuning_qwen_emb', 'checkpoint-8000')
    # Train the model
    trainer.train()
    
    ########## Inference ##########
    print(compute_metrics(trainer.predict(test_dataset), join(output_dir, 'test.json')))