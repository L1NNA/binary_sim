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
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.nn import DataParallel
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, recall_score, precision_score

from data_loaders.causal_dataset import CausalDataset, causal_collate, CausalDatasetPair, causal_pair_collate
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, LoraConfig

from utils import compute_metrics
import os
import sentencepiece as spm
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
from models.causalLM import CustomModelCausal, CustomConfig

from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig

from transformers import Cache, DynamicCache

from torchsummary import summary

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--batch_size", type=int, default=6,
        help="The batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=8,
        help="The batch size."
    )
    parser.add_argument(
        "--max_lines", type=int, default=0,
        help="max lines"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=200_000,
        help="maxinum number of pairs to sample that form causal sentences"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50_000,
        help="max vocabulary size"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
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
        "--hidden_size", type=int, default=768,
        help="hidden size of the model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=16,
        help="number of attention layers"
    )
    parser.add_argument(
        "--kv_heads", type=int, default=2,
        help="number of kv heads for GQA"
    )
    parser.add_argument(
        "--q_heads", type=int, default=12,
        help="number of q heads"
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048,
        help="intermediate size for FFN"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="dropout"
    )
    parser.add_argument(
        "--use_blk_mask", action='store_true',
        help="whether to use <blk> mask for additional structure learning"
    )
    # Parse the arguments
    args = parser.parse_args()

if __name__ == "__main__":

    ################# load the dataset #################
    path = os.path.join(os.getcwd(), 'datasets')
    path = os.path.join(path, 'train.jsonl')
    dataset = CausalDatasetPair(path, max_line=args.max_lines, max_num = args.max_pairs)

    total_length = len(dataset)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length
    # split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])


    ################# Train a sentencepiece tokenizer #################
    tok_path = tokenizer_path = os.path.join(os.getcwd(), 'tokenizers')
    if not os.path.exists(tok_path):
        os.makedirs(tok_path)
        tokenizer = SentencePieceBPETokenizer()
        special_tokens = ['<addr>', '<byte>', '<str>', '<BLK>']

        tokenizer_training_text = dataset.get_all()

        tokenizer.train_from_iterator(
            tokenizer_training_text,
            vocab_size=args.vocab_size,
            min_frequency=5,
            show_progress=True,
            limit_alphabet=500,
            special_tokens = special_tokens,
        )

        transformer_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer)
        transformer_tokenizer.add_special_tokens({'pad_token': '<PAD>'})  # Add pad token
        transformer_tokenizer.add_special_tokens({'eos_token': '<EOS>'}) # Add eos token
        transformer_tokenizer.add_special_tokens({'sep_token': '<SEP>'}) # sep token
        transformer_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        transformer_tokenizer.save_pretrained(tok_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_path, model_max_length = args.max_length, padding_side='left', truncation_side='left')

    tokenizer.add_special_tokens({'pad_token': '<PAD>'})  # Add pad token
    tokenizer.add_special_tokens({'eos_token': '<EOS>'}) # Add eos token
    tokenizer.add_special_tokens({'sep_token': '<SEP>'}) # sep token
    tokenizer.add_special_tokens({'unk_token': '<unk>'})

    special_token_length = len(tokenizer.special_tokens_map)
    vocab_size = tokenizer.vocab_size
    vocab_size = vocab_size+special_token_length
    # try pretrained tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m', trust_remote_code=True, padding_side='left', truncation_side='left')
    # vocab_size = tokenizer.vocab_size
    
    ################# define a custom LM ##############################

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    training_config = CustomConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        kv_heads = args.kv_heads,
        num_attention_heads=args.q_heads,
        intermediate_size=args.d_ff,
        dropout = args.dropout,
        device = device,
        dtype = 'bfloat16',
        use_cache=False,
        causal_lm = True,
        max_len = args.max_length,
        layer_norm_eps = 1e-5,
        padding_idx = tokenizer.pad_token_id,
        output_attentions = False,
        output_hidden_states = False,
        return_dict = True,
        use_flash_attn = False,
        use_alibi = True,
        use_blk_mask = args.use_blk_mask
    )

    model = CustomModelCausal(training_config).to(device, dtype=torch.bfloat16)

    # ### generate some text
    # all_text = dataset.get_all()
    # input = tokenizer(all_text[0], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length, return_token_type_ids = False).to(device)
    # outputs = model(**input, max_new_tokens=150, use_cache=True, past_key_values=DynamicCache())
    

    # print(summary(model))

    ################# Train the model ##############################
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f'./model_checkpoints/causal_custom'
    warmup_steps = int(1000/args.batch_size)
    training_args = TrainingArguments(
        use_cpu=True if device ==  'cpu' else False,
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        # gradient_checkpointing=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        # gradient_accumulation_steps=4,
        lr_scheduler_type = "cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 2},
        weight_decay=0.01,
        learning_rate=0.0001,
        logging_dir=join(output_dir, 'logs', current_time),
        logging_steps=250,
        evaluation_strategy='epoch',     # Evaluate at the end of each epoch
        save_strategy='epoch',
        # save_steps=500,
        load_best_model_at_end=True, # Load the best model when finished training
        metric_for_best_model='loss'
    )

    # Define the Trainer
    # metric_wrapper = lambda p: compute_metrics(p, join(output_dir, 'valid.jsonl'))
    collator = lambda x: causal_pair_collate(x, tokenizer, args.max_lines, args.max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)

    ########## Inference ##########
    # print(compute_metrics(trainer.predict(test_dataset), join(output_dir, 'test.json')))


    ### example of using DynamicCache to store past key values for generation
    ''' from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        # Prepare a cache class and pass it to model's forward
        past_key_values = DynamicCache()
        outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
    '''