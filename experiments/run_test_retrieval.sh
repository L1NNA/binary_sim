export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

# python run_test_retrieval.py \
#     --model 'codet5p-110m-embedding' --local_model_path "./model_checkpoints/simcse_codet5p-110m-embedding/checkpoint-1608" \
#     --max_length 512 --test_batch_size 32

# python run_test_retrieval.py \
#     --model 'qwen_llm2vec' --local_model_path "./model_checkpoints/simcse_codeqwen2vec/checkpoint-5625" \
#     --max_length 512 --test_batch_size 8
    
# python run_test_retrieval.py \
#     --model 'qwen_emb' \
#     --max_length 1024 --test_batch_size 128 --instruct 1

    
python run_test_retrieval.py \
    --model 'qwen_emb' \
    --max_length 512 --test_batch_size 32 --note "zero-shot"