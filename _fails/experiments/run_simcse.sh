export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

# torchrun --nproc_per_node=8 --master_port=1234 run_simcse.py --model 'graphcodebert' \
#     --max_length 512 --batch_size 12
    
# torchrun --nproc_per_node=8 --master_port=1234 run_simcse.py --model 'codet5p-110m-embedding' \
#     --max_length 512 --batch_size 12 --resume

# python run_simcse.py --model 'qwen_emb' \
#     --max_length 512 --batch_size 2 --epochs 1

# torchrun --nproc_per_node=7 --master_port=1234
# torchrun --nproc_per_node=4 --master_port=1234 run_simcse.py \
#     --model 'codeqwen2vec' --local_model_path "./model_checkpoints/mntp_codeqwen/checkpoint-55804"\
#     --max_length 512 --batch_size 4 --note "test" --epochs 3 --data "train"


python run_simcse.py \
    --model 'qwen_emb' --embedding_loss 'info_nce' \
    --batch_size 2