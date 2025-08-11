import subprocess
import os
import argparse
from typing import List, Dict, Any, Optional, Set


# structure: { "model_name": { "path": "mode_path", "type": "model_type", "checkpoints": [check_point number] } }
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # --- with_think model ---
    "Qwen3-14B-5epoch-v0": {
        "path": "", # your model path
        "type": "with_think",
        "checkpoints": [540]
    },
}

DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    "with_think": {
        "val": "./dataset/val_dataset_t.jsonl",
        "test": "./dataset/test_sft_input_only_no_think.jsonl",
    },
}


SEEDS_TO_RUN: List[int] = [42] 

PYTHON_SCRIPT_PATH = "./capture_logits.py"

BASE_OUTPUT_DIR = "./logits_data_seed"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 16384
CUDA_VISIBLE_DEVICES = "0"



def run_capture_for_checkpoint(
    model_name: str,
    model_base_dir: str,
    dataset_split: str,
    dataset_path: str,
    output_dir: str,
    cp: Optional[int],
    seed: int  
):
    if cp is not None:
        model_path = f"{model_base_dir}/checkpoint-{cp}-merged"
        run_identifier = f"cp{cp}"
        print(f"=============== Processing Fine-tuned Model: '{model_name}', Checkpoint: {cp}, Dataset: '{dataset_split}', Seed: {seed} ===============")
    else:
        model_path = model_base_dir
        run_identifier = "pretrained"
        print(f"=============== Processing Pre-trained Model: '{model_name}', Dataset: '{dataset_split}', Seed: {seed} ===============")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model path not found: {model_path}")
        print(f"Skipping this task.")
        return False

    file_basename = f"{model_name}_{run_identifier}_{dataset_split}_seed{seed}"
    metadata_file = f"{output_dir}/{file_basename}_metadata.jsonl"
    logits_file = f"{output_dir}/{file_basename}_logits.npz"

    bash_command = f"""
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}

    python {PYTHON_SCRIPT_PATH} \\
        --model "{model_path}" \\
        --val_dataset "{dataset_path}" \\
        --metadata_path "{metadata_file}" \\
        --logits_output_path "{logits_file}" \\
        --batch_size {BATCH_SIZE} \\
        --max_new_tokens {MAX_NEW_TOKENS} \\
        --seed {seed}
    """

    print("Executing Command:\n---")
    print(bash_command.strip())
    print("---\n")

    try:
        subprocess.run(bash_command, shell=True, check=True, text=True)
        print(f"✅ Successfully processed: '{file_basename}'")

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: An error occurred while processing '{file_basename}'.")
        print(f"Return Code: {e.returncode}")
        return False
    
    except FileNotFoundError:
        print(f"❌ ERROR: Command not found. Is '{PYTHON_SCRIPT_PATH}' a valid path?")
        return False

    return True


def main():
    model_choices = list(MODEL_CONFIG.keys())
    model_type_choices = list(set(details['type'] for details in MODEL_CONFIG.values()))
    
    parser = argparse.ArgumentParser(
        description="Run logit capture for different models, checkpoints, and datasets.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "--model_selection",
        type=str,
        nargs='+',
        required=True,
        help=(
            "Select models to run. Accepts one or more values.\n"
            "Choices include:\n"
            "- 'all': Run all configured models.\n"
            f"- A model type: {model_type_choices}\n"
            f"- A specific model name: {model_choices}"
        )
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="all",
        choices=['test', 'val', 'all'],
        help="The dataset split to run on: 'test', 'val', or 'all' (default: 'all')."
    )
    args = parser.parse_args()

    selected_models: Set[str] = set()
    if 'all' in args.model_selection:
        selected_models.update(MODEL_CONFIG.keys())
    else:
        for selection in args.model_selection:
            if selection in MODEL_CONFIG:
                selected_models.add(selection)
            elif selection in model_type_choices:
                for name, details in MODEL_CONFIG.items():
                    if details['type'] == selection:
                        selected_models.add(name)
            else:
                print(f"Warning: Selection '{selection}' is not a valid model name or type. It will be ignored.")

    if not selected_models:
        print("❌ No valid models selected to run. Exiting.")
        return

    if args.run_mode == 'all':
        dataset_splits_to_run = ['val', 'test']
    else:
        dataset_splits_to_run = [args.run_mode]

    print(f"[*] Seeds to run: {SEEDS_TO_RUN}")
    print(f"[*] Models to run: {sorted(list(selected_models))}")
    print(f"[*] Dataset splits to run: {dataset_splits_to_run}\n")
    
    for seed in SEEDS_TO_RUN:
        for model_name in sorted(list(selected_models)):
            model_details = MODEL_CONFIG[model_name]
            model_base_dir = model_details['path']
            model_type = model_details['type']
            checkpoints = model_details['checkpoints']

            # Create independent output based on the model names
            model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
            os.makedirs(model_specific_output_dir, exist_ok=True)

            print(f"\n--- Starting tasks for model: '{model_name}' ---")
            print(f"[*] Using Seed: {seed}")
            print(f"[*] Model Type: '{model_type}'")
            print(f"[*] Output will be saved to: '{model_specific_output_dir}'")

            for dataset_split in dataset_splits_to_run:
                if model_type not in DATASET_CONFIG or dataset_split not in DATASET_CONFIG[model_type]:
                    print(f"❌ ERROR: No dataset path configured for model_type='{model_type}' and split='{dataset_split}'. Skipping.")
                    continue
                
                dataset_path = DATASET_CONFIG[model_type][dataset_split]

                for cp in checkpoints:
                    success = run_capture_for_checkpoint(
                        model_name=model_name,
                        model_base_dir=model_base_dir,
                        dataset_split=dataset_split,
                        dataset_path=dataset_path,
                        output_dir=model_specific_output_dir,
                        cp=cp,
                        seed=seed 
                    )
                    if not success:
                        print(f"\nHalting script due to an error in the last step for model '{model_name}' with seed {seed}.")
                        return 
    
    print("\n================= All selected tasks finished for all seeds. =================\n")


if __name__ == "__main__":
    main()