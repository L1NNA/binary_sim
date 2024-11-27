export CUDA_VISIBLE_DEVICES=0,1,4,5
torchrun --nproc_per_node=4 --master_port=1234 run_simcse.py --model 'graphcodebert' \
    --max_length 512 --batch_size 8