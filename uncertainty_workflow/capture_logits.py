"""
Environment: Python 3.10
ms-swift==3.6.2
transformers==4.52.4
torch==2.5.1+cu121
flash-attn==2.7.1.post1
outlines==1.1.1
outlines-core==0.1.26
"""
import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from typing import List, Dict, Any
from tqdm import tqdm

import outlines
from outlines.processors.guide import RegexGuide

class StatefulDynamicProcessor(LogitsProcessor):
    def __init__(self, tokenizer, log_container: Dict, eos_token_id: int):
        self.tokenizer = tokenizer
        self.log_container = log_container
        self.eos_token_id = eos_token_id
        self.search_regex = r'(?:<think>[\s\S]*?</think>\s*)?<JSON>\s*\{\s*\"policy_stance_label\"\s*:\s*\"'
        self.search_guide = RegexGuide.from_regex(self.search_regex, self.tokenizer)
        self.reset()
        print("✅ StatefulDynamicProcessor initialized for single-file logit capture.")

    def set_prompt_lengths(self, lengths: List[int]):
        self.prompt_lengths = {i: lengths[i] for i in range(len(lengths))}

    def reset(self, batch_start_index: int = 0):
        # Reset internal states and set the starting index for the new batch
        self.states = {}
        self.guide_states = {}
        self.log_container.clear()
        self.prompt_lengths = {}
        self.batch_start_index = batch_start_index

    def _initialize_state_for_batch(self, batch_size):
        if not self.states:
            self.states = {i: "SEARCHING" for i in range(batch_size)}
            self.guide_states = {i: self.search_guide.initial_state for i in range(batch_size)}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        self._initialize_state_for_batch(batch_size)
        
        for i in range(batch_size):
            state = self.states.get(i)
            if state == "DONE":
                continue

            if state == "SEARCHING":
                prompt_len = self.prompt_lengths.get(i, 0)
                if input_ids.shape[1] <= prompt_len:
                    continue

                last_token = input_ids[i, -1].item()
                current_guide_state = self.guide_states[i]
                next_guide_state = self.search_guide.get_next_state(current_guide_state, last_token)
                self.guide_states[i] = next_guide_state
                
                if self.search_guide.is_final_state(next_guide_state):
                    global_item_idx = self.batch_start_index + i
                    print(f"\n[INFO] Global Item {global_item_idx} (Batch local index {i}): Regex matched. Capturing logits tensor.")
                    
                    self.log_container[i] = {"decision_logits_tensor": scores[i].cpu()}
                    self.states[i] = "DONE"
                    mask = torch.full_like(scores[i], -float('inf'))
                    mask[self.eos_token_id] = 0.0
                    scores[i] = mask
                else:
                    allowed_tokens = self.search_guide.get_next_instruction(next_guide_state).tokens
                    mask = torch.full_like(scores[i], -float('inf'))
                    mask[allowed_tokens] = scores[i, allowed_tokens]
                    scores[i] = mask
        
        return scores

def get_data_batches(data: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main(args):
    # Stage 1: Loading model and tokenizer
    if args.seed is not None:
        print(f"--> Using fixed random seed: {args.seed}. Forcing sampling count to 1.")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    print("--> Stage 1: Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2"
    )
    raw_tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side='left'
    )
    if raw_tokenizer.pad_token is None:
        raw_tokenizer.pad_token = raw_tokenizer.eos_token
    eos_token_id = raw_tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must have a defined `eos_token_id`.")
    tokenizer_for_outlines = outlines.models.transformers.TransformerTokenizer(raw_tokenizer)
    model.eval()

    # Stage 2: Preparing data 
    print("--> Stage 2: Preparing data...")
    with open(args.val_dataset, 'r', encoding='utf-8') as f:
        prompts_data = []
        for idx, line in enumerate(f):
            item = json.loads(line)
            item['id'] = f'item_{idx:06d}'
            prompts_data.append(item)
    print(f"Dataset loaded with {len(prompts_data)} entries, each assigned a unique ID.")
    
    # Initialization of processor and lists
    batch_capture_logs = {}
    logits_capture_processor = StatefulDynamicProcessor(
        tokenizer=tokenizer_for_outlines, log_container=batch_capture_logs, eos_token_id=eos_token_id
    )
    logits_processor_list = LogitsProcessorList([logits_capture_processor])

    print("--> Stage 3: Starting logits capture process...")
    results_metadata = []
    all_logits_for_saving = {} 
    num_batches = (len(prompts_data) + args.batch_size - 1) // args.batch_size
    data_generator = get_data_batches(prompts_data, args.batch_size)
    
    pbar = tqdm(enumerate(data_generator), total=num_batches, desc="Capturing Logits")

    for batch_idx, batch in pbar:
        initial_prompts = [raw_tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=True) for item in batch]
        inputs = raw_tokenizer(initial_prompts, return_tensors="pt", padding=True).to(model.device)
        
        input_token_lengths = [len(ids) for ids in inputs.input_ids]

        batch_start_index = batch_idx * args.batch_size
        logits_capture_processor.reset(batch_start_index=batch_start_index)
        
        logits_capture_processor.set_prompt_lengths(input_token_lengths)
        
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=raw_tokenizer.eos_token_id, logits_processor=logits_processor_list
            )
        
        for i in range(len(batch)):
            item_id = batch[i]['id']
            metadata_item = {"id": item_id, "original_messages": batch[i]['messages']}
            capture_log = batch_capture_logs.get(i)
            
            if capture_log and "decision_logits_tensor" in capture_log:
                all_logits_for_saving[item_id] = capture_log["decision_logits_tensor"].numpy()
            
            results_metadata.append(metadata_item)

    # Stage 4: Saving all data to files
    print("\n--> Stage 4: Saving all data to files...")
    print(f"Saving {len(all_logits_for_saving)} logit arrays to {args.logits_output_path}...")
    np.savez_compressed(args.logits_output_path, **all_logits_for_saving)
    
    with open(args.metadata_path, 'w', encoding='utf-8') as f:
        for item in results_metadata:
            f.write(json.dumps(item) + '\n')
            
    print(f"Logits capture complete.")
    print(f"Metadata saved to: {args.metadata_path}")
    print(f"All logits saved in one file: {args.logits_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture logits efficiently into a single NPZ file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--val_dataset", type=str, required=True, help="Path to the validation dataset (.jsonl).")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path for the output metadata file (.jsonl).")
    parser.add_argument("--logits_output_path", type=str, required=True, help="Path for the single output .npz file for all logits.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size. Reduce if OOM.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to search for the regex pattern.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    
    main(args)