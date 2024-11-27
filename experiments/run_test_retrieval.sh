export CUDA_VISIBLE_DEVICES=2

# python run_test_retrieval.py \
#     --model 'codet5p-110m-embedding' --local_model_path "./model_checkpoints/simcse_codet5p-110m-embedding/checkpoint-1608" \
#     --max_length 512 --test_batch_size 32

# python run_test_retrieval.py \
#     --model 'qwen_llm2vec' --local_model_path "./model_checkpoints/simcse_codeqwen2vec/checkpoint-5625" \
#     --max_length 512 --test_batch_size 8
    
python run_test_retrieval.py \
    --model 'qwen_emb' \
    --max_length 512 --test_batch_size 8 --instruct 1

    
# python run_test_retrieval.py \
#     --model 'graphcodebert' --local_model_path "./model_checkpoints/simcse_graphcodebert/checkpoint-7035" \
#     --max_length 512 --test_batch_size 32