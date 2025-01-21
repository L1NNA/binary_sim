# Binary Sim

## Setup

Use conda to manage environments
```shell script
conda create -n bin_sim python=3.9
conda activate bin_sim
# Install pytorch with cuda
pip install -r requirements.txt
```

## Load Dataset
Download the test files from the release page, place under `datasets` folder
If you're curious about the data generation process, check the files under folder `data_gen`.

## All components

### 1-2. Data augmentation and Causal Training
Please check the causal generation process in `data_loaders\causal_dataset.py` 
```bash
torchrun --nproc_per_node=4 --master_port=1234 run_causalLM.py \
    --model codeqwen  \
    --data train --batch_size 2 --max_length 1024 \
    --epochs 2
```

### 3. LLM2Vec
See `models\llm2vec.py` for the model setup
```bash
torchrun --nproc_per_node=4 --master_port=1234 run_custom_mntp.py \
    --model='codeqwen' --local_model_path [causal_checkpoint] \
    --max_seq_len 512 --epochs=2
```

### 4. Contrastive learning (same process for all baselines)
See `layers\loss.py` for all the losses
```bash
torchrun --nproc_per_node=4 --master_port=1234 run_simcse.py \
    --model 'codeqwen2vec' --local_model_path [mntp_checkpoint] \
    --max_length 512 --batch_size 4 --note "test" --epochs 3 --data "train"
```

### Test and evaluation
Run `run_test_retrieval[_BinCorp].py` for retrieval tasks

Run `run_t_SNE.py` to plot the embedding distributions

