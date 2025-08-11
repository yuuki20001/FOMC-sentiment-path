
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

swift infer \
    --model "your model" \
    --infer_backend pt \
    --max_batch_size 32 \
    --temperature 0.0 \
    --max_new_tokens 16384 \
    --val_dataset "./dataset/test/test_sft_input_only_no_think.jsonl" \
    --result_path "your path" \
    --attn_impl flash_attn \