#!/usr/bin/env bash

# Batch inference helper for stance-uncertainty experiments using the Qwen3-14B model
# (5-epoch LoRA adapter). Quickstart:
#   1. Activate the environment with torch, transformers, and outlines.
#   2. Update the --model/--val_dataset/--result_path arguments to your local checkpoints and datasets.
#   3. Adjust CUDA_VISIBLE_DEVICES to match your hardware before running `bash uncertainty_workflow/stance_uncertainty_infer_qwen3-14B-5epoch.sh`.
# The script intentionally avoids storing credentials or proprietary paths; review arguments before sharing artifacts.

# Reduce CUDA allocator fragmentation for long-context decoding. Remove if unsupported on your system.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Pin the run to specific devices so multi-user servers remain predictable.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch batch decoding with uncertainty-aware control. Edit CLI values to suit your experiment.
python ./uncertainty_workflow/stance_uncertainty_infer_batch.py \
    --model "your model path" \
    --val_dataset "./dataset/test/test_sft_input_only_no_think.jsonl" \
    --result_path "./output/Qwen3-14B-5epoch-lora-seed_42-K_10-threshold_08-can-neutral.jsonl" \
    --max_new_tokens 16384 \
    --batch_size 32 \
    --logtoku_k 10 \
    --uncertainty_method 'method3' \
    --aggressive_strategy "greedy_candidate" \
    --high_uncertainty_strategy "neutral" \
    --uncertainty_threshold 0.03560897707939148 \
    --seed 42
