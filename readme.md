# Overview
Fedspeak—the stylized discourse used in Federal Open Market Committee (FOMC) communications—carries implicit policy signals that steer global markets. This repository accompanies our paper and releases an LLM-based, uncertainty-aware pipeline that ingests FOMC transcripts and classifies policy stance (hawkish/neutral/dovish) while logging the reasoning paths.

we augment the input texts by extracting financial entity relations and reasoning over monetary policy transmission paths using structured templates. We introduce a dynamic uncertainty decoding module with a PU metric that helps identify potentially unreliable predictions, aiming to improve overall prediction reliability. 

# Core environment:
Python 3.10.18

ms-swift==3.6.2

transformers==4.52.4

torch==2.5.1+cu121

flash-attn==2.7.1.post1

outlines==1.1.1

outlines-core==0.1.26

# how to use sft_workflow
Before running the ``.sh`` script, please make sure that the ms-swift framework and all required dependencies (including flash-attention, deepzero, etc.) have been properly installed. 
After installation, fill in the corresponding file paths in the script as needed.
Then you can run ``bash train_Qwen3-14B-lora-no_think_step_save.sh``.

# How to use Uncertainty workflow
Configure the following parameters in batch_capture_search_seed.py:
- Path to the input data file
- Path to the model
- path to the output file
- Random seed
- Dataset type & model type

run ``python batch_capture_search_seed.py --model_selection all --run_mode all``

Note:
``--model_selection all`` for all model in "MODEL_CONFIG", if you want to run single model, use: ``--model_selection Qwen3-14B-5epoch-v0``

if only run "Qwen3-14B-5epoch-v0", "test set", run ``python batch_capture_search_seed.py --model_selection Qwen3-14B-5epoch-v0 --run_mode test``

After running the script, it will generate the corresponding logits and metadata files.

You should set the hyperparameter configuration in ``hyper_para_search.py``.

Then, fill in the paths to these files in ``run_search.sh`` and update the input file path(logits, metadata), model path, and output path accordingly. Running this script ( ``bash run_search.sh``) will produce the hyperparameter search results and the final statistical report.

# Stance uncertainty inference
- `uncertainty_workflow/stance_uncertainty_infer_batch.py`: Batch decoder that loads the model, applies regex-guided JSON generation, and computes uncertainty scores; tune `--val_dataset`, `--result_path`, `--logtoku_k`, `--uncertainty_method`, plus the strategy/threshold flags before running.
- `uncertainty_workflow/stance_uncertainty_infer_qwen3-14B-5epoch.sh`: Convenience launcher for the Qwen3-14B 5-epoch LoRA checkpoint; it exports CUDA env vars and forwards the recommended CLI arguments. Update the `--model`, dataset, and output paths to match your environment before running.

