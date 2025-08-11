# Supplemantary material file structure:
├── dataset/                              
│   ├── train/                           
│   ├── val/                             
│   └── test/                            
├── sft_workflow/                         
│   ├── train_Qwen3-14B-lora-no_think_step_save.sh
│   └── infer_Qwen3-14B_lora_5epoch_no_think_step_pt.sh
├── uncertainty_workflow/                 
│   ├── hyper_para_search.py             
│   ├── capture_logits.py                
│   ├── batch_capture_search_seed.py     
│   └── run_search.sh                    
├── GT_data/                             
├── finalizer_sentiment_prompt.md

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
``--model_selection all`` for all model in "MODEL_CONFIG", if you want to run single model, like: ``--model_selection Qwen3-14B-5epoch-v0``
if only run Qwen3-14B-5epoch-v0, test stage, run ``python batch_capture_search_seed.py --model_selection Qwen3-14B-5epoch-v0 --run_mode test``

After running the script, it will generate the corresponding logits and metadata files.
You should set the hyperparameter configuration in ``hyper_para_search.py``.
Then, fill in the paths to these files in ``run_search.sh`` and update the input file path(logits, metadata), model path, and output path accordingly. Running this script ( ``bash run_search.sh``) will produce the hyperparameter search results and the final statistical report.
