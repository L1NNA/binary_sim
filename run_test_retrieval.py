import json
import argparse
from datetime import datetime
from os.path import join
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,5'

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer, TrainerCallback
from torch.utils.data import Dataset, DataLoader, random_split

from models.coe_cos_sim import CoECosSim
from models.embedding_model import QwenEmbeddingModel
from models.prefix_sft import PrefixSFT, Sanity
# from data_loaders.pos_neg_bin_sim_dataset import BinSimDataset, line_collate
from data_loaders.test_retrieval_dataset import RetrievalDataset
from utils import find_common_elements, calculate_mrr, compute_retrieval_metrics, get_tokens
from utils.retrieval_utils import get_embeddings
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, LoraConfig
from safetensors.torch import load_file
from models.llm2vec import Qwen2BiForMNTP

models = {
    'qwen_emb': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'codet5p-110m-embedding': 'Salesforce/codet5p-110m-embedding',
    'codet5p-220m': "Salesforce/codet5p-220m",
    'codet5p-770m': "Salesforce/codet5p-770m",
    'jina_emb': 'jinaai/jina-embeddings-v2-base-code',
    'qwen_llm2vec': 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
}

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="The batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32,
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
    parser.add_argument(
        "--source_dataset", type=str, default='o0',
        help="source dataset for retrieval"
    )
    parser.add_argument(
        "--target_dataset", type=str, default='o2',
        help="target dataset for retrieval"
    )
    parser.add_argument(
        "--pool_size", type=int, default=1000,
        help="pool size for retrieval"
    )
    parser.add_argument(
        # "--sft", type=str, default='lora',
        "--sft", type=str, default=None,
        help="whether to use sft model or pre-trained model"
    )
    # Parse the arguments
    args = parser.parse_args()
    
    
    ## Results folder based on vanilla model or 
    if args.sft is not None:
        source_embs_path = f'./results/{args.sft}_{args.model}_{args.source_dataset}_{args.max_length}_{args.pool_size}.jsonl'
        target_embs_path = f'./results/{args.sft}_{args.model}_{args.target_dataset}_{args.max_length}_{args.pool_size}.jsonl'
    else:
        source_embs_path = f'./results/baseline_{args.model}_{args.source_dataset}_{args.max_length}_{args.pool_size}.jsonl'
        target_embs_path = f'./results/baseline_{args.model}_{args.target_dataset}_{args.max_length}_{args.pool_size}.jsonl'
        
        
    if os.path.exists(source_embs_path) and os.path.exists(target_embs_path):
        query_embs = []
        value_embs = []
        sampled_keys = []
        with open(source_embs_path) as sf:
            for line in sf:
                js = json.loads(line)
                query_embs.append(js['embs'])
                sampled_keys.append(js['func'])
        with open(target_embs_path) as tf:
            for i, line in enumerate(tf):
                js = json.loads(line)
                value_embs.append(js['embs'])
                assert js['func'] == sampled_keys[i]
                
        query_embs = torch.tensor(query_embs)
        value_embs = torch.tensor(value_embs)
    else:
        # ================= Load Data ======================
        ## read in all function keys from source and target
        path = os.path.join(os.getcwd(), 'datasets')
        source_path = os.path.join(path, 'test_'+args.source_dataset+'.jsonl')
        target_path = os.path.join(path, 'test_'+args.target_dataset+'.jsonl')
        source_dataset = RetrievalDataset(source_path, keys=None)
        target_dataset = RetrievalDataset(target_path, keys=None)

        # ================= Sample Data ======================
        ## sample pool_size number of functions that exist in both source and target
        source_keys = list(source_dataset.bins['function'])
        target_keys = list(target_dataset.bins['function'])
        common_keys = find_common_elements([source_keys, target_keys])
        assert len(common_keys) >= args.pool_size, "pool size too large, not enough functions with this setup"
        sampled_keys = random.sample(common_keys, args.pool_size) 

        source_funcs = source_dataset.get_all(keys=sampled_keys, join_blocks=(args.max_lines==0))
        target_funcs = target_dataset.get_all(keys=sampled_keys, join_blocks=(args.max_lines==0))


        # ================= Load Backbone ======================
        if args.model == 'qwen_llm2vec':
            backbone = Qwen2BiForMNTP.from_pretrained(
                './model_checkpoints/mntp_codeqwen/checkpoint-7848',
                torch_dtype='auto',
            ).to('cuda:0')
        else:
            backbone = AutoModel.from_pretrained(
                models[args.model],
                torch_dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(
            models[args.model],
            trust_remote_code=True,
            padding_side='left'
        )
        
        # ================= Load Custom Lora Model ======================
        if args.sft is not None:
            ckp = os.path.join(os.getcwd(), 'model_checkpoints', f'PrefixTuning_{args.model}', 'checkpoint-13334', 'model.safetensors')
            
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
            
            state_dict = load_file(ckp)
            model.load_state_dict(state_dict)
        else:
            model = backbone

        # ================= Embedding ======================
        
        ## describe prompt for both source and target
        task_source = f'Given an assembly, retrieve the clone'
        task_target = f'Given an assembly, retrieve the clone'
        
        def get_output(input_ids, attention_mask, sft = False):
            if sft == True:
                preds = model(input_ids, attention_mask)
                preds['preds'] = preds['preds'].float()
                return preds
            ### check if [batch, seq_len] or [batch, blocks, seq_len]
            
            if args.model == 'qwen_llm2vec':
                preds = model(input_ids, attention_mask, output_hidden_states=True)
                return {"preds":preds['hidden_states'][-1][:, -1, :].float()}
            if input_ids.dim() == 2:
                preds = model(input_ids, attention_mask)
                if type(preds) is torch.Tensor:
                    preds = preds.float()
                else:
                    preds = preds.last_hidden_state[:, -1, :].float()
                return {"preds":preds}
            else:
                block_pred = []
                for i in range(input_ids.size(1)):
                    block_pred.append(model(input_ids[:, i, :], attention_mask[:, i, :]).last_hidden_state[:, -1, :].float())
                block_pred = torch.stack(block_pred, dim=0)
                preds = torch.mean(block_pred, dim=0)
                return {"preds":preds}
        query_embs, value_embs = get_embeddings(source_funcs, target_funcs, 
                                                get_output, 
                                                tokenizer, args.max_length, 
                                                test_batch_size=args.test_batch_size,
                                                max_blocks=args.max_lines,
                                                task_source=task_source, task_target=task_target,
                                                sft = True if args.sft is not None else False
                                                )

        # Cache outputs
        with open(source_embs_path, 'a') as sf, open(target_embs_path, 'a') as tf:
            for i in range(len(sampled_keys)):
                sf.write(json.dumps({'func': sampled_keys[i], 'embs': query_embs[i].numpy().tolist()}) + '\n')
                tf.write(json.dumps({'func': sampled_keys[i], 'embs': value_embs[i].numpy().tolist()}) + '\n')

    ### compute cosine similarity
    scores = []
    for i in query_embs:
        scores.append(F.cosine_similarity(i.unsqueeze(0), value_embs, dim=1).numpy())
    scores = np.array(scores)
    
    ## this takes too much memory for large pool size like 10k
    # scores = F.cosine_similarity(query_embs.unsqueeze(1), value_embs.unsqueeze(0), dim=2).numpy()
    relevance = np.arange(query_embs.size(0))
    
    print(compute_retrieval_metrics(scores, relevance))
    
    
    
    
    