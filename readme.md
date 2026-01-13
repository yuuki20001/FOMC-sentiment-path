# FOMC-sentiment-path

## Overview

This repository accompanies our paper *Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths* (AAAI 2026) and provides an end-to-end codebase for:

- Deciphering Fedspeak and classifying the underlying monetary policy stance (`HAWKISH` / `NEUTRAL` / `DOVISH`) with an LLM-based pipeline.
- Enriching inputs with domain-specific reasoning grounded in the monetary policy transmission mechanism (via structured prompt templates).
- Running uncertainty-aware decoding to quantify prediction confidence and support downstream analysis (including scripts for logits capture and hyperparameter search).

## Repository Structure

- `dataset/`: Datasets and intermediate artifacts (e.g., `.jsonl`).
- `prompt_template/`: Structured prompt templates for input construction/enrichment.
- `sft_workflow/`: Supervised Fine-Tuning (SFT) workflows (based on `ms-swift`).
- `uncertainty_workflow/`: Uncertainty workflows (logits capture, hyperparameter search, uncertainty-aware inference).
- `GT_data/`: Ground-truth annotations and evaluation data.


## Requirements

The project is typically run in a conda environment named `fin_path`. A reference set of core dependencies used by the `uncertainty_workflow/` scripts:

- Python 3.10
- `ms-swift==3.6.2`
- `transformers==4.52.4`
- `torch==2.5.1+cu121`
- `flash-attn==2.7.1.post1`
- `outlines==1.1.1`
- `outlines-core==0.1.26`

## Quick Start

1) Create and activate an environment (example)

```bash
conda create -n fin_path python=3.10 -y
conda activate fin_path
```
2) Supervised Fine-Tuning (SFT) workflow

The `sft_workflow/` directory contains shell launchers for training/inference/merging LoRA adapters using the `ms-swift` framework. Please follow the official installation instructions in the [`ms-swift`](https://github.com/modelscope/ms-swift), then edit paths inside the scripts before running them:

- `sft_workflow/train_Qwen3-14B-lora-no_think_step_save.sh`
- `sft_workflow/infer_Qwen3-14B_lora_5epoch_no_think_step_pt.sh`
- `sft_workflow/merge_lora_adapter.sh`

3) Logits capture + hyperparameter search (for threshold/strategy analysis)

- Run `uncertainty_workflow/batch_capture_search_seed.py`  (needs to edit in-file config):

```bash
python batch_capture_search_seed.py --model_selection all --run_mode all
```
Note: `--model_selection all` for all model in "MODEL_CONFIG"

4) Uncertainty-aware stance inference

```bash
python uncertainty_workflow/stance_uncertainty_infer_batch.py \
  --model "your model path" \
  --val_dataset "./dataset/test/test_sft_input_only_no_think.jsonl" \
  --result_path "./output/results.jsonl" \
  --batch_size 8 \
  --max_new_tokens 1024 \
  --logtoku_k 25 \
  --uncertainty_method method1 \
  --aggressive_strategy greedy_candidate \
  --high_uncertainty_strategy cluster_sampling \
  --uncertainty_threshold 0.02
```

Note: `--uncertainty_threshold` is an absolute uncertainty score cutoff, not a ratio. In our workflow it is typically selected from a percentile of the empirical uncertainty distribution (estimated via the hyperparameter search stage), then used as a fixed numeric threshold at inference time.

To reproduce a recommended configuration, edit and run:

```bash
bash uncertainty_workflow/stance_uncertainty_infer_qwen3-14B-5epoch.sh
```



- Configure the search space in `uncertainty_workflow/hyper_para_search.py`, then fill paths in `uncertainty_workflow/run_search.sh` and run:

```bash
bash uncertainty_workflow/run_search.sh
```



## Disclaimer

This repository is provided for research and educational purposes only. It does not constitute financial, investment, legal, or tax advice, and should not be relied upon for making trading or investment decisions. The authors and contributors make no representations or warranties regarding the accuracy, completeness, or suitability of the information or any outputs produced by the code. Use of this repository is at your own risk; you are solely responsible for complying with applicable laws and regulations and for any decisions or actions taken based on the results.

## Citation
```python
@article{yao2025interpreting,
  title={Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths},
  author={Yao, Rui and Chai, Qi and Yao, Jinhai and Li, Siyuan and Chen, Junhao and Zhang, Qi and Wang, Hao},
  journal={arXiv preprint arXiv:2508.08001},
  year={2025}
}
```
