export CUDA_VISIBLE_DEVICES=0,4,5

python run_mntp.py \
    --model 'codeqwen' \
    --max_seq_len 1024 --batch_size 8 --note "join" --epochs 2