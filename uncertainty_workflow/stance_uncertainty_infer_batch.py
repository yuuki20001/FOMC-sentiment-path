import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.special import digamma
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from typing import List, Dict, Any, Set
import itertools
import outlines
from outlines.processors.guide import RegexGuide


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"--> Random seed set to {seed} for reproducibility.")



def cal_ea_and_cr(evidence_2d: np.ndarray, k_values: np.ndarray, use_log_smoothing: bool) -> (np.ndarray, np.ndarray):
    """
    Compute EA and CR from the provided evidence tensor and the requested top-k sizes.
    Intended to run after tensors move from GPU to CPU.
    """
    alpha = evidence_2d + 1
    alpha_0 = np.sum(alpha, axis=1, keepdims=True)
    
    digamma_alpha_plus_1 = digamma(alpha + 1)
    digamma_alpha_0_plus_1 = digamma(alpha_0 + 1)
    ea_scores = -np.sum((alpha / alpha_0) * (digamma_alpha_plus_1 - digamma_alpha_0_plus_1), axis=1)
    
    total_positive_evidence = np.sum(evidence_2d, axis=1, keepdims=True)
    
    if use_log_smoothing:
        log_smoothed_k = np.log1p(k_values)
        cr_scores = (log_smoothed_k / (total_positive_evidence.flatten() + log_smoothed_k))
    else:
        cr_scores = (k_values / alpha_0.flatten())

    return ea_scores, cr_scores


class StatefulDynamicProcessor(LogitsProcessor):
    """
    Stateful LogitsProcessor that batches GPU-side uncertainty estimates and enforces deterministic decoding decisions.
    """
    def __init__(self, tokenizer, label_token_ids: Dict[str, int], args: argparse.Namespace, device: torch.device, log_container: Dict, model: AutoModelForCausalLM):
        self.tokenizer = tokenizer
        self.raw_tokenizer = tokenizer.tokenizer
        
        self.vocab_size = model.config.vocab_size 
        
        self.label_token_ids = label_token_ids
        self.candidate_labels = list(label_token_ids.keys()) # Order: HAWKISH, NEUTRAL, DOVISH
        self.label_to_int = {label: i for i, label in enumerate(self.candidate_labels)}
        self.args = args
        self.device = device
        self.log_container = log_container
        
        self.search_regex = r'(?:<think>[\s\S]*?</think>\s*)?<JSON>\s*\{\s*\"policy_stance_label\"\s*:\s*\"'
        self.search_guide = RegexGuide.from_regex(self.search_regex, self.tokenizer)

        self._initialize_label_groups()
        self.reset()
        print(f"✅ StatefulDynamicProcessor initialized with BATCH-OPTIMIZED uncertainty calculation.")

    def _generate_case_combinations(self, word: str) -> set:
        if not word: return set()
        char_options = [(c.lower(), c.upper()) for c in word]
        return {"".join(combo) for combo in itertools.product(*char_options)}

    def _initialize_label_groups(self):
        label_stems = {
            "HAWKISH": ["hawkish", "hawk", "hawks", "haw", "ha", "h"],
            "NEUTRAL": ["neutral", "neut", "neu", "ne", "n"],
            "DOVISH": ["dovish", "dove", "dov", "do", "d"]
        }
        base_words_from_stems = {label: set() for label in label_stems}
        for label, stems in label_stems.items():
            for stem in stems:
                base_words_from_stems[label].update(self._generate_case_combinations(stem))
        label_source_words = {}
        for label, words_set in base_words_from_stems.items():
            expanded_words = set()
            for word in words_set:
                expanded_words.add(word); expanded_words.add(" " + word)
            label_source_words[label] = list(expanded_words)

        self.token_map = torch.full((self.vocab_size,), -1, dtype=torch.long, device=self.device)
        self.token_id_to_label_group_cpu = {} # For CPU-side final mapping
        for label, words in label_source_words.items():
            label_idx = self.label_to_int[label]
            for word in words:
                token_ids = self.raw_tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 1:
                    token_id = token_ids[0]
                    self.token_map[token_id] = label_idx
                    if token_id not in self.token_id_to_label_group_cpu:
                         self.token_id_to_label_group_cpu[token_id] = label

    def set_prompt_lengths(self, lengths: List[int]):
        self.prompt_lengths = {i: lengths[i] for i in range(len(lengths))}

    def reset(self):
        self.states = {}
        self.guide_states = {}
        self.enforcement_queues = {}
        self.log_container.clear()
        self.prompt_lengths = {}

    def _initialize_state_for_batch(self, batch_size):
        if not self.states:
            self.states = {i: "SEARCHING" for i in range(batch_size)}
            self.guide_states = {i: self.search_guide.initial_state for i in range(batch_size)}
            self.enforcement_queues = {i: [] for i in range(batch_size)}

    def _calculate_uncertainties_in_batch(self, logits_batch: torch.FloatTensor) -> Dict[str, Any]:
        """
        Calculates EA and CR for a batch, ensuring logic matches the analysis script.
        Final uncertainty score is EA * CR.
        method1 = log-smoothed logits; method2 = same aggregation without smoothing;
        method3 = aggregated-label logits concatenated with unclassified logits prior to EA*CR.
        """
        num_samples = logits_batch.shape[0]
        k = self.args.logtoku_k
        results = {}
        

        positive_logits = logits_batch.clamp(min=0)
        
        top_k_logits, top_k_indices = torch.topk(positive_logits, k, dim=-1)
        top_k_labels = self.token_map[top_k_indices]

        all_evidence_m12, all_k_m12 = [], []
        aggregated_top_k_list = []
        for i in range(num_samples):
            aggr = torch.zeros(3, device=self.device)
            is_unclassified = (top_k_labels[i] == -1)
            
            for label_idx in range(3):
                mask = (top_k_labels[i] == label_idx)
                if mask.any():
                    aggr[label_idx] = top_k_logits[i, mask].sum()
            
            uncl = top_k_logits[i, is_unclassified]
            evidence = torch.cat([aggr, uncl]).cpu().numpy()

            all_evidence_m12.append(evidence)
            all_k_m12.append(len(evidence))
            aggregated_top_k_list.append(aggr.cpu().numpy())

        max_len = max(all_k_m12) if all_k_m12 else 0
        m12_evidence_np = np.array([np.pad(e, (0, max_len - len(e))) for e in all_evidence_m12])
        m12_k_np = np.array(all_k_m12)
        
        if m12_evidence_np.size > 0:
            ea_m1, cr_m1 = cal_ea_and_cr(m12_evidence_np, m12_k_np, use_log_smoothing=True)
            results['uncert_score_method1'] = cr_m1 * ea_m1
            results['ea_method1'], results['cr_method1'] = ea_m1, cr_m1
            
            ea_m2, cr_m2 = cal_ea_and_cr(m12_evidence_np, m12_k_np, use_log_smoothing=False)
            results['uncert_score_method2'] = cr_m2 * ea_m2
            results['ea_method2'], results['cr_method2'] = ea_m2, cr_m2
        
        results['aggregated_logits_top_k'] = np.array(aggregated_top_k_list)
        
        num_labels = len(self.candidate_labels)
        scatter_map_labels_only = self.token_map.clone()
        aggregated_labels = torch.zeros((num_samples, num_labels), device=self.device)
        
        for label_idx in range(num_labels):
            mask = (scatter_map_labels_only == label_idx)
            aggregated_labels[:, label_idx] = (positive_logits * mask).sum(dim=1)

        
        unclassified_mask = (self.token_map == -1)
        unclassified_logits = positive_logits * unclassified_mask

        # Concatenate aggregated logits for the 3 labels with all unclassified logits
        m3_candidate_logits = torch.cat([aggregated_labels, unclassified_logits], dim=1)

        # Select Top-K entries from the concatenated vector using hyperparameter k
        # K refers to the hyperparameter, not the vector length itself
        # Handle cases where k exceeds the vector length
        actual_k = min(k, m3_candidate_logits.shape[1])
        m3_top_k_evidence_tensor, _ = torch.topk(m3_candidate_logits, actual_k, dim=1)
        m3_top_k_evidence_np = m3_top_k_evidence_tensor.cpu().numpy()
        
        # Compute final uncertainty scores
        m3_k = actual_k
        ea_m3, cr_m3 = cal_ea_and_cr(m3_top_k_evidence_np, np.full(num_samples, m3_k), use_log_smoothing=False)

        results['uncert_score_method3'] = cr_m3 * ea_m3
        results['ea_method3'], results['cr_method3'] = ea_m3, cr_m3

        return results

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        self._initialize_state_for_batch(batch_size)
        
        decided_indices, guide_masks = [], {}

        # --- Step 1: State Update and Collection ---
        for i in range(batch_size):
            state = self.states.get(i)
            if state == "DONE" or state == "ENFORCING": continue

            if state == "SEARCHING":
                prompt_len = self.prompt_lengths.get(i, 0)
                if input_ids.shape[1] <= prompt_len: continue
                
                last_token = input_ids[i, -1].item()
                next_guide_state = self.search_guide.get_next_state(self.guide_states[i], last_token)
                self.guide_states[i] = next_guide_state
                
                if self.search_guide.is_final_state(next_guide_state):
                    self.states[i] = "DECIDED"
                    decided_indices.append(i)
                else:
                    allowed_tokens = self.search_guide.get_next_instruction(next_guide_state).tokens
                    mask = torch.full_like(scores[i], -float('inf'))
                    mask[allowed_tokens] = scores[i, allowed_tokens]
                    guide_masks[i] = mask

        for i, mask in guide_masks.items(): scores[i] = mask
            
        # --- Step 2: Batch Uncertainty Calculation for all DECIDED items ---
        if decided_indices:
            logits_for_calc = scores[decided_indices]
            batch_uncertainty_results = self._calculate_uncertainties_in_batch(logits_for_calc)

            # --- Step 3: Apply Decisions and Logging for the sub-batch ---
            aggregated_logits_top_k_np = batch_uncertainty_results.get('aggregated_logits_top_k', np.array([]))
            
            for j, original_index in enumerate(decided_indices):
                uncertainty_score = batch_uncertainty_results[f'uncert_score_{self.args.uncertainty_method}'][j]

                # Retain auxiliary EA*CR diagnostics for downstream logging and analysis
                ea_score = batch_uncertainty_results[f'ea_{self.args.uncertainty_method}'][j]
                cr_score = batch_uncertainty_results[f'cr_{self.args.uncertainty_method}'][j]

                aggregated_logits_top_k = {label: aggregated_logits_top_k_np[j, i] for i, label in enumerate(self.candidate_labels)}
                
                chosen_label_str, decision_strategy_info = "", ""

                if uncertainty_score > self.args.uncertainty_threshold:
                    strategy = self.args.high_uncertainty_strategy
                    decision_strategy_info = f"Conservative ({strategy})"
                    if strategy == 'neutral':
                        chosen_label_str = "NEUTRAL"
                    elif strategy == 'cluster_sampling' or strategy == 'candidate_sampling':
                        if strategy == 'cluster_sampling':
                            aggr_tensor = torch.from_numpy(aggregated_logits_top_k_np[j]).to(self.device)
                            top_2_logits, top_2_indices = torch.topk(aggr_tensor, 2)
                        else: # candidate_sampling
                            top_2_logits, top_2_indices = torch.topk(logits_for_calc[j], 2)

                        probs = F.softmax(top_2_logits / self.args.temperature_for_label, dim=-1)
                        sampled_relative_idx = torch.multinomial(probs, 1).item()
                        
                        if strategy == 'cluster_sampling':
                           chosen_label_str = self.candidate_labels[top_2_indices[sampled_relative_idx].item()]
                        else:
                           sampled_token_id = top_2_indices[sampled_relative_idx].item()
                           chosen_label_str = self.token_id_to_label_group_cpu.get(sampled_token_id, "NEUTRAL")
                else:
                    strategy = self.args.aggressive_strategy
                    decision_strategy_info = f"Aggressive ({strategy})"
                    if strategy == 'greedy_k_aggregate':
                        chosen_label_str = max(aggregated_logits_top_k, key=aggregated_logits_top_k.get)
                    elif strategy == 'greedy_candidate':
                        chosen_token_id = torch.argmax(logits_for_calc[j]).item()
                        chosen_label_str = self.token_id_to_label_group_cpu.get(chosen_token_id, "NEUTRAL")

                self.enforcement_queues[original_index] = self.raw_tokenizer.encode(chosen_label_str, add_special_tokens=False)
                self.log_container[original_index] = {
                    "uncertainty_calculation_method": self.args.uncertainty_method,
                    "decision_uncertainty": float(uncertainty_score),
                    "aleatoric_uncertainty": float(ea_score),
                    "epistemic_uncertainty": float(cr_score),
                    "final_decision_path": decision_strategy_info,
                    "chosen_label_at_decision": chosen_label_str,
                    "aggregated_logits_strength": {k: float(v) for k,v in aggregated_logits_top_k.items()},
                }
                self.states[original_index] = "ENFORCING"

        # --- Step 4: Apply Enforcement Masks for all ENFORCING items ---
        for i in range(batch_size):
            if self.states[i] == "ENFORCING":
                queue = self.enforcement_queues[i]
                if not queue:
                    self.states[i] = "DONE"
                    continue
                
                required_token_id = queue.pop(0)
                mask = torch.full_like(scores[i], -float('inf'))
                mask[required_token_id] = 0.0
                scores[i] = mask
        
        return scores


# Main Logic

def get_data_batches(data: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main(args):
    # Set the random seed at the beginning of the main function if provided.
    if args.seed is not None:
        set_seed(args.seed)

    print("--> Stage 1: Loading model and tokenizer...")
    # ... (Rest of the main function is unchanged) ...
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2"
        )
    except Exception:
        print("Warning: Flash Attention 2 not available. Loading model without it.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True
        )

    raw_tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side='left'
    )
    if raw_tokenizer.pad_token is None:
        raw_tokenizer.pad_token = raw_tokenizer.eos_token
    tokenizer_for_outlines = outlines.models.transformers.TransformerTokenizer(raw_tokenizer)
    model.eval()

    print("--> Stage 2: Preparing data and components...")
    print("--> Stage 2: Preparing data and components...")
    with open(args.val_dataset, 'r', encoding='utf-8') as f:
        prompts_data = [json.loads(line) for line in f]
    print(f"Dataset loaded with {len(prompts_data)} entries.")
    
    label_token_ids = {
        "HAWKISH": raw_tokenizer.encode("HAWKISH", add_special_tokens=False)[0],
        "NEUTRAL": raw_tokenizer.encode("NEUTRAL", add_special_tokens=False)[0],
        "DOVISH": raw_tokenizer.encode("DOVISH", add_special_tokens=False)[0]
    }

    batch_decision_logs = {}
    
    dynamic_processor = StatefulDynamicProcessor(tokenizer_for_outlines, label_token_ids, args, model.device, batch_decision_logs, model)    
    logits_processor_list = LogitsProcessorList([dynamic_processor])

    print("--> Stage 3: Starting integrated dynamic generation...")
    results = []
    num_batches = (len(prompts_data) + args.batch_size - 1) // args.batch_size
    data_generator = get_data_batches(prompts_data, args.batch_size)
    
    for batch in tqdm(data_generator, total=num_batches, desc="Dynamic Inference"):
        initial_prompts = [raw_tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=True) for item in batch]
        inputs = raw_tokenizer(initial_prompts, return_tensors="pt", padding=True).to(model.device)
        
        input_token_lengths = [len(ids) for ids in inputs.input_ids]
        dynamic_processor.reset()
        dynamic_processor.set_prompt_lengths(input_token_lengths)
        
        with torch.no_grad():
            generated_outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature_for_text if args.do_sample else None,
                pad_token_id=raw_tokenizer.eos_token_id, 
                logits_processor=logits_processor_list
            )
        
        for i in range(len(batch)):
            input_length = input_token_lengths[i]
            generated_token_ids = generated_outputs[i][input_length:]
            clean_generated_text = raw_tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            result_item = {"original_messages": batch[i]['messages']}
            decision_log = batch_decision_logs.get(i, {})
            result_item.update(decision_log)
            result_item["generated_text"] = clean_generated_text
            results.append(result_item)

    print("\n--> Stage 4: Saving results...")
    result_dir = os.path.dirname(args.result_path)
    if result_dir: os.makedirs(result_dir, exist_ok=True)
    with open(args.result_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Inference complete. Results saved to: {args.result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible, Batch-Optimized dynamic decoding with selectable uncertainty and strategy.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--val_dataset", type=str, required=True, help="Path to the validation dataset (.jsonl).")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the output results (.jsonl).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If set, this will fix seeds for torch and numpy.")
    
    parser.add_argument("--logtoku_k", type=int, default=25, help="Uncertainty Param; Top-K tokens for aggregation.")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5, help="Decision Param; Uncertainty threshold.")
    parser.add_argument("--uncertainty_method", type=str, default="method1", choices=['method1', 'method2', 'method3'], help="Selects uncertainty scoring recipe: method1=log-smoothed agg logits, method2=non-smoothed, method3=concat agg+unclassified logits.")
    parser.add_argument("--aggressive_strategy", type=str, default="greedy_candidate", choices=['greedy_candidate', 'greedy_k_aggregate'], help="Low Uncertainty Strategy.")
    parser.add_argument("--high_uncertainty_strategy", type=str, default="cluster_sampling", choices=['neutral', 'cluster_sampling', 'candidate_sampling'], help="High Uncertainty Strategy.")
    parser.add_argument("--temperature_for_label", type=float, default=0.7, help="Sampling Param; Temperature for label sampling.")
    parser.add_argument("--do_sample", action='store_true', help="Enable sampling for explanatory text generation.")
    parser.add_argument("--temperature_for_text", type=float, default=0.7, help="Temperature for text generation.")
    
    args = parser.parse_args()
    main(args)