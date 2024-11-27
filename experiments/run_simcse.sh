# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# torchrun --nproc_per_node=4 --master_port=1234 run_simcse.py --model 'graphcodebert' \
#     --max_length 512 --batch_size 8
    
torchrun --nproc_per_node=8 --master_port=1234 run_simcse.py --model 'codet5p-110m-embedding' \
    --max_length 512 --batch_size 12 --resume

# torchrun --nproc_per_node=8 --master_port=1234 run_simcse.py \
#     --model 'codeqwen2vec' --local_model_path "./model_checkpoints/mntp_codeqwen/checkpoint-7848" \
#     --max_length 512 --batch_size 3