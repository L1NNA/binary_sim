export CUDA_VISIBLE_DEVICES=0,1,4,5
python run_test_retrieval.py \
    --model 'graphcodebert' --local_model_path "./model_checkpoints/simcse_graphcodebert/checkpoint-7035"\
    --max_length 512 --test_batch_size 32