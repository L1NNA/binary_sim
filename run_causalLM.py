import os
import argparse
from datetime import datetime
from os.path import join

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from torch.utils.data import random_split

from data_loaders.causal_dataset import CausalDatasetPair, causal_pair_collate



models = {
    'codet5p-220m': "Salesforce/codet5p-220m",
    'codet5p-770m': "Salesforce/codet5p-770m",
    'codeqwen': 'Qwen/Qwen2.5-Coder-0.5B-Instruct',
}


def main():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="The batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=2,
        help="The batch size."
    )
    parser.add_argument(
        "--data", type=str, default='train_BinaryCorp'
    )
    parser.add_argument(
        "--model", choices=models.keys(), default='codeqwen',
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
    parser.add_argument(
        "--max_pairs", type=int, default=100000,
        help="max number of tokens per line"
    )
    parser.add_argument(
        "--resume", action='store_true',
        help="resume"
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="number of epochs"
    )
    # Parse the arguments
    args = parser.parse_args()

    
    ################# Load PLM ##############################
    checkpoint = models[args.model]

    if args.model == 'codeqwen':
        backbone = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        backbone = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, padding_side='left', truncation_side='left')


    ########## Load Data ##########
    path = os.path.join(os.getcwd(), 'datasets')
    path = os.path.join(path, f'{args.data}.jsonl')
    dataset = CausalDatasetPair(path, max_line=args.max_lines, max_num=args.max_pairs)
    total_length = len(dataset)
    train_length = int(total_length * 0.9)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    
    
    
    ########## Load Model #########
    special_tokens = ['<addr>', '<byte>', '<str>', '<BLK>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.eos_token = '<EOS>'
    tokenizer.sep_token = '<SEP>'
    tokenizer.pad_token = '<PAD>'
    tokenizer.unk_token = '<unk>'
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # device = 'cpu'
    
    model = backbone.to(device)
    
    # ckp = os.path.join(os.getcwd(), 'model_checkpoints', f'causal_{args.model}', 'checkpoint-8672', 'model.safetensors')
    # state_dict = load_file(ckp)
    # state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']
    # model.load_state_dict(state_dict)
    
    # lora_modules = []
    # for n, m in backbone.named_modules():
    #     if type(m) is nn.modules.linear.Linear and 'mlp' in n:
    #         lora_modules.append(n)

    # lora_config = LoraConfig(
    #     # target_modules=[r"model.layers.%d.mlp.gate_proj", r"model.layers.%d.mlp.up_proj", r"model.layers.%d.mlp.down_proj"]
    #     # target_modules=r'.*\.mlp\.%s_proj'
    #     target_modules = lora_modules
    # )

    # lora_model = get_peft_model(backbone, lora_config)
    # lora_model.print_trainable_parameters()

    ########## Training ##########
    # Define the training arguments
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f'./model_checkpoints/causal_{args.model}_{args.data}'
    warmup_steps = int(1000/args.batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        # gradient_checkpointing=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        learning_rate=1e-5,
        logging_dir=join(output_dir, 'logs', current_time),
        logging_steps=250,
        # eval_steps=1000,
        # eval_strategy="steps",
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
        # compute_metrics=metric_wrapper,
    )
    
    
    

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)

    ########## Inference ##########
    # print(compute_metrics(trainer.predict(test_dataset), join(output_dir, 'test.json')))

if __name__ == "__main__":
    main()