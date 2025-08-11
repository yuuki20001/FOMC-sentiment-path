"""
Environment: Python 3.10
ms-swift==3.6.2
transformers==4.52.4
torch==2.5.1+cu121
flash-attn==2.7.1.post1
outlines==1.1.1
outlines-core==0.1.26
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, confusion_matrix
import argparse
import json
import torch.nn.functional as F
from scipy.special import digamma
from transformers import AutoTokenizer
from typing import Dict, Any, List
import itertools
import statsmodels.api as sm
from scipy.stats import ttest_ind, mannwhitneyu
from datetime import datetime

class DecisionAnalysisHelper:
    def __init__(self, tokenizer):
        self.raw_tokenizer = tokenizer
        self.candidate_labels = ["HAWKISH", "NEUTRAL", "DOVISH"]
        self.label_map = {label: i for i, label in enumerate(self.candidate_labels)}
        self.unmapped_tokens = set() 
        self._initialize_label_groups()

    def _generate_case_combinations(self, word: str) -> set:
        if not word: return set()
        char_options = [(c.lower(), c.upper()) for c in word]
        return {"".join(combo) for combo in itertools.product(*char_options)}

    def _initialize_label_groups(self):
        label_stems = {"HAWKISH": ["hawkish", "hawk", "hawks", "haw", "ha", "h"], "NEUTRAL": ["neutral", "neut", "neu", "ne", "n"], "DOVISH": ["dovish", "dove", "dov", "do", "d"]}
        base_words_from_stems = {label: set() for label in label_stems}
        for label, stems in label_stems.items():
            for stem in stems:
                base_words_from_stems[label].update(self._generate_case_combinations(stem))
        label_source_words = {}
        for label, words_set in base_words_from_stems.items():
            expanded_words = set()
            for word in words_set:
                expanded_words.add(word)
                if len(word) > 0: expanded_words.add(" " + word)
            label_source_words[label] = list(expanded_words)
        self.label_group_ids: dict[str, set[int]] = {label: set() for label in label_source_words}
        self.token_id_to_label_group: dict[int, str] = {}
        for label, words in label_source_words.items():
            for word in words:
                token_ids = self.raw_tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 1:
                    token_id = token_ids[0]
                    self.label_group_ids[label].add(token_id)
                    if token_id not in self.token_id_to_label_group: self.token_id_to_label_group[token_id] = label


def cal_ea_and_cr(logits: np.ndarray, k: int, use_log_smoothing: bool = False) -> (float, float):
    evidence = logits
    if evidence.ndim == 1: evidence_2d = np.array([evidence])
    else: evidence_2d = evidence
    evidence_2d = np.maximum(evidence_2d, 0)
    alpha = evidence_2d + 1
    alpha_0 = np.sum(alpha, axis=1, keepdims=True)
    term_1 = digamma(alpha + 1)
    term_2 = digamma(alpha_0 + 1)
    ea_result = -np.sum((alpha / alpha_0) * (term_1 - term_2), axis=1)
    total_positive_evidence = np.sum(evidence_2d, axis=1, keepdims=True)
    if use_log_smoothing:
        log_smoothed_k = np.log1p(k)
        cr_score = log_smoothed_k / (total_positive_evidence + log_smoothed_k)
    else:
        cr_score = (k / alpha_0)
    if evidence.ndim == 1:
        return ea_result[0], cr_score[0][0]
    return ea_result, cr_score.flatten()


def calculate_uncertainties_batched(
    all_logits: torch.FloatTensor,
    analysis_helper: DecisionAnalysisHelper,
    k: int
) -> Dict[str, np.ndarray]:
    num_samples, vocab_size = all_logits.shape
    device = all_logits.device
    results = {}

    label_to_int = {label: i for i, label in enumerate(analysis_helper.candidate_labels)}
    token_map = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    for label, group_ids in analysis_helper.label_group_ids.items():
        if len(group_ids) > 0:
            token_map[list(group_ids)] = label_to_int[label]
    
    positive_logits = all_logits.clamp(min=0)
    token_map_scatter = token_map.clone()
    token_map_scatter[token_map_scatter == -1] = 3 
    aggregated_all_labels = torch.zeros((num_samples, 4), device=device)
    index_for_scatter = token_map_scatter.unsqueeze(0).expand(num_samples, -1)
    aggregated_all_labels.scatter_add_(1, index_for_scatter, positive_logits)
    
    m3_base_evidence = aggregated_all_labels[:, :3].cpu().numpy()
    unclassified_mask = (token_map == -1)
    unclassified_logits = positive_logits * unclassified_mask 
    m3_candidate_logits = np.concatenate([m3_base_evidence, unclassified_logits.cpu().numpy()], axis=1)
    m3_top_k_evidence = -np.partition(-m3_candidate_logits, k, axis=1)[:, :k]
    m3_ea, m3_cr = cal_ea_and_cr(m3_top_k_evidence, k, use_log_smoothing=False)
    results["uncert_3"] = m3_cr * m3_ea
    
    return results


def calculate_eval_metrics(y_true, y_pred):
    label_order = ["HAWKISH", "NEUTRAL", "DOVISH"]
    if not y_true or not y_pred:
        metrics = {"macro_f1": 0.0, "weighted_f1": 0.0, "precision": 0.0}
        cm = np.zeros((len(label_order), len(label_order)), dtype=int)
        return metrics, cm
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=label_order, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=label_order, zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', labels=label_order, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    metrics = {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "precision": precision}
    return metrics, cm


def map_prediction_to_canonical_label(
    prediction_token_id: int, 
    analysis_helper: DecisionAnalysisHelper
) -> str:
    label = analysis_helper.token_id_to_label_group.get(prediction_token_id)
    if label is not None:
        return label
    else:
        analysis_helper.unmapped_tokens.add(prediction_token_id)
        return "NEUTRAL"


def determine_prediction(
    row: pd.Series, 
    strategy: str, 
    temperature: float, 
    analysis_helper: DecisionAnalysisHelper
) -> str:
    if strategy == "NEUTRAL":
        return "NEUTRAL"
    elif strategy == "cluster_sampling":
        aggr_logits_dict = row['aggregated_logits']
        sorted_labels = sorted(aggr_logits_dict.keys(), key=lambda k: aggr_logits_dict[k], reverse=True)
        if not sorted_labels: return "NEUTRAL"
        top_2_labels = sorted_labels[:2]
        top_2_logits = torch.tensor([aggr_logits_dict[label] for label in top_2_labels])
        if len(top_2_logits) == 1: return top_2_labels[0]
        probs = F.softmax(top_2_logits / temperature, dim=-1)
        sampled_idx = torch.multinomial(probs, 1).item()
        return top_2_labels[sampled_idx]
    elif strategy == "candidate_sampling":
        logits_tensor = row['logits_tensor']
        top_2_logits, top_2_indices = torch.topk(logits_tensor, 2)
        probs = F.softmax(top_2_logits / temperature, dim=-1)
        sampled_relative_idx = torch.multinomial(probs, 1).item()
        sampled_token_id = top_2_indices[sampled_relative_idx].item()
        return map_prediction_to_canonical_label(sampled_token_id, analysis_helper)
    return "NEUTRAL"


def main():
    # Hyperparameter Configuration
    HYPER_K_VALUES = [3, 10, 15, 20, 25, 30]
    THRESHOLD_PERCENTILES = [1,0.95,0.9,0.85,0.8,0.75,0.7]
    CONSERVATIVE_STRATEGIES = ["NEUTRAL", "cluster_sampling", "candidate_sampling"]
    SAMPLING_COUNT = 1
    TEMPERATURE_VALUES =[0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run comprehensive uncertainty and performance evaluation.")
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--logits_file_path", type=str, required=True)
    parser.add_argument("--output_base_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If provided, sets the seed and forces sampling count to 1 (e.g., --seed 42).")
    args = parser.parse_args()

    # Initial seed setting
    if args.seed is not None:
        print(f"--> Using fixed random seed: {args.seed}. Forcing sampling count to 1.")
        SAMPLING_COUNT = 1
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Phase 1: Loading and Pre-calculating All Greedy Predictions
    print("--> Phase 1: Loading and Pre-calculating All Greedy Predictions...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    analysis_helper = DecisionAnalysisHelper(tokenizer)
    with open(args.metadata_path, 'r', encoding='utf-8') as f: metadata_list = [json.loads(line) for line in f]
    logits_archive = np.load(args.logits_file_path)
    with open(args.ground_truth_path, 'r', encoding='utf-8') as f: ground_truth_data = json.load(f)
    
    text_to_ground_truth = {entry["input_entry"]["sentence"]: entry["input_entry"]["mapped_label"] for entry in ground_truth_data}
    
    unified_data = []
    logits_map = {item_id: torch.from_numpy(logits_archive[item_id]) for item_id in logits_archive.files}

    for metadata_item in tqdm(metadata_list, desc="Phase 1: Matching and Pre-calculating"):
        user_content = next((msg.get("content", "") for msg in metadata_item.get("original_messages", []) if msg.get("role") == "user"), "")
        original_text_match = re.search(r'1\.\s+\*\*original_text:\*\*\s+```json\n\s*(.*?)\n\s*```', user_content, re.DOTALL)
        if not original_text_match: continue
        
        original_text = original_text_match.group(1).strip()
        ground_truth_label = text_to_ground_truth.get(original_text)
        item_id = metadata_item['id']

        if ground_truth_label and item_id in logits_map:
            logits_tensor = logits_map[item_id]
            positive_logits = logits_tensor.clamp(min=0)
            
            top1_token_id = torch.argmax(logits_tensor).item()
            greedy_candidate_label = map_prediction_to_canonical_label(top1_token_id, analysis_helper)

            aggregated_logits_full = {label: 0.0 for label in analysis_helper.candidate_labels}
            for token_id, logit_val in enumerate(positive_logits):
                label_group = analysis_helper.token_id_to_label_group.get(token_id)
                if label_group: aggregated_logits_full[label_group] += logit_val.item()

            greedy_K_aggregate_labels = {}
            for k in HYPER_K_VALUES:
                top_k_logits, top_k_indices = torch.topk(positive_logits, k)
                
                aggregated_logits_k = {label: 0.0 for label in analysis_helper.candidate_labels}
                for i in range(len(top_k_indices)):
                    token_id = top_k_indices[i].item()
                    logit_val = top_k_logits[i].item()
                    label_group = analysis_helper.token_id_to_label_group.get(token_id)
                    if label_group:
                        aggregated_logits_k[label_group] += logit_val
                
                greedy_K_aggregate_labels[k] = max(aggregated_logits_k, key=aggregated_logits_k.get) if aggregated_logits_k else "NEUTRAL"

            unified_data.append({
                "id": item_id,
                "ground_truth_label": ground_truth_label,
                "logits_tensor": logits_tensor,
                "aggregated_logits": aggregated_logits_full,
                "greedy_candidate": greedy_candidate_label,
                "greedy_K_aggregate": greedy_K_aggregate_labels
            })
    
    print(f"Successfully loaded and pre-calculated for {len(unified_data)} data points.")


    # Phase 2: Calculating Uncertainty Scores
    print("\n--> Phase 2: Calculating Uncertainty Scores...")
    uncertainty_records = []
    if unified_data:
        all_logits_tensor = torch.stack([item['logits_tensor'] for item in unified_data])
        for k_value in tqdm(HYPER_K_VALUES, desc="Processing K values"):
            batched_scores = calculate_uncertainties_batched(all_logits_tensor, analysis_helper, k_value)
            for i, item in enumerate(unified_data):
                record = {
                    'id': item['id'], 'k_value': k_value, 
                    'ground_truth_label': item['ground_truth_label'],
                    'logits_tensor': item['logits_tensor'],
                    'aggregated_logits': item['aggregated_logits'],
                    'greedy_candidate': item['greedy_candidate'],
                    'greedy_K_aggregate': item['greedy_K_aggregate'][k_value],
                    'uncert_3': batched_scores['uncert_3'][i],
                }
                uncertainty_records.append(record)
    uncertainty_df = pd.DataFrame(uncertainty_records)

    #Phase 3: Generating Threshold Report
    print("\n--> Phase 3: Generating Threshold Report...")
    threshold_report = {}
    if not uncertainty_df.empty:
        uncert_col = 'uncert_3'
        if uncert_col in uncertainty_df.columns:
            threshold_stats = uncertainty_df.groupby('k_value')[uncert_col].quantile(THRESHOLD_PERCENTILES).unstack()
            method_thresholds = {f"HYPER-K={k}": v for k, v in threshold_stats.to_dict("index").items()}
            threshold_report[f'method_3_thresholds'] = method_thresholds
    
    report_path = f"{args.output_base_path}_thresholds_{timestamp}.json"
    with open(report_path, 'w') as f: json.dump(threshold_report, f, indent=4)
    print(f"Threshold summary report saved to: {report_path}")

    #Phase 4: Simulating Strategies and Calculating Performance
    print("\n--> Phase 4: Simulating Strategies and Calculating Performance...")
    evaluation_results = []
    detailed_predictions_log = {}

    BASE_GREEDY_STRATEGIES = ['greedy_candidate', 'greedy_K_aggregate']
    UNCERT_METHOD = 'uncert_3'

    if not uncertainty_df.empty:
        pbar_total = len(BASE_GREEDY_STRATEGIES) * len(HYPER_K_VALUES) * len(THRESHOLD_PERCENTILES) * len(CONSERVATIVE_STRATEGIES) * len(TEMPERATURE_VALUES)
        pbar = tqdm(total=pbar_total, desc="Evaluating Strategies")

        for base_greedy_strategy in BASE_GREEDY_STRATEGIES:
            for k_value in HYPER_K_VALUES:
                for percentile in THRESHOLD_PERCENTILES:
                    k_specific_df = uncertainty_df[uncertainty_df['k_value'] == k_value]
                    if k_specific_df.empty:
                        pbar.update(len(CONSERVATIVE_STRATEGIES) * len(TEMPERATURE_VALUES))
                        continue
                    
                    absolute_threshold = -1.0 if percentile == 0.0 else k_specific_df[UNCERT_METHOD].quantile(percentile)
                    y_true = k_specific_df['ground_truth_label'].tolist()
                    y_ids = k_specific_df['id'].tolist() 

                    for temperature in TEMPERATURE_VALUES:
                        for cons_strategy in CONSERVATIVE_STRATEGIES:
                            if cons_strategy == "NEUTRAL" and temperature != TEMPERATURE_VALUES[0]:
                                pbar.update(1)
                                continue

                            # for each combination of hyperparameters is independent and repeatable.
                            if args.seed is not None:
                                np.random.seed(args.seed)
                                torch.manual_seed(args.seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed(args.seed)
                                    torch.cuda.manual_seed_all(args.seed)

                            temp_str = f"t={temperature}" if cons_strategy != "NEUTRAL" else "t=N/A"
                            hyperparam_key_str = (f"{base_greedy_strategy}|k={k_value}|{UNCERT_METHOD}|"
                                                  f"p={percentile}|{cons_strategy}|{temp_str}")
                            detailed_predictions_log[hyperparam_key_str] = {}
                            
                            eval_row = {
                                "base_greedy_strategy": base_greedy_strategy,
                                "conservative_strategy": cons_strategy, 
                                "k_value": k_value, 
                                "uncert_method": UNCERT_METHOD, 
                                "threshold_percentile": percentile, 
                                "temperature": temperature if cons_strategy != "NEUTRAL" else "N/A"
                            }

                            if cons_strategy in ["cluster_sampling", "candidate_sampling"] and SAMPLING_COUNT > 1:
                                trial_metrics_list, trial_cms_list = [], []
                                all_trials_y_pred = [] 
                                
                                for _ in range(SAMPLING_COUNT):
                                    y_pred_trial = []
                                    for _, row in k_specific_df.iterrows():
                                        if row[UNCERT_METHOD] >= absolute_threshold:
                                            prediction = determine_prediction(row, cons_strategy, temperature, analysis_helper)
                                        else:
                                            prediction = row[base_greedy_strategy]
                                        y_pred_trial.append(prediction)
                                    
                                    metrics_trial, cm_trial = calculate_eval_metrics(y_true, y_pred_trial)
                                    trial_metrics_list.append(metrics_trial)
                                    trial_cms_list.append(cm_trial)
                                    all_trials_y_pred.append(y_pred_trial) 

                                df_trials = pd.DataFrame(trial_metrics_list)
                                avg_metrics = df_trials.mean().to_dict()
                                best_trial_idx = df_trials['macro_f1'].idxmax()
                                best_metrics = df_trials.loc[best_trial_idx].to_dict()
                                best_y_pred = all_trials_y_pred[best_trial_idx]
                                for i, sample_id in enumerate(y_ids):
                                    detailed_predictions_log[hyperparam_key_str][sample_id] = best_y_pred[i]
                                
                                eval_row.update({f"avg_{k}": v for k, v in avg_metrics.items()})
                                eval_row.update({f"best_{k}": v for k, v in best_metrics.items()})
                                eval_row['confusion_matrix'] = np.sum(trial_cms_list, axis=0).tolist()
                            else: 
                                y_pred_list = []
                                for _, row in k_specific_df.iterrows():
                                    if row[UNCERT_METHOD] > absolute_threshold:
                                        prediction = determine_prediction(row, cons_strategy, temperature, analysis_helper)
                                    else:
                                        prediction = row[base_greedy_strategy]
                                    y_pred_list.append(prediction)
                                    detailed_predictions_log[hyperparam_key_str][row['id']] = prediction
                                
                                metrics, cm = calculate_eval_metrics(y_true, y_pred_list)
                                eval_row.update(metrics)
                                eval_row.update({f"best_{k}": v for k, v in metrics.items()}) 
                                eval_row['confusion_matrix'] = cm.tolist()
                            
                            evaluation_results.append(eval_row)
                            pbar.update(1)
        pbar.close()

    # --- Phase 5: Performing Statistical Correlation Analysis ---
    print("\n--> Phase 5: Performing Statistical Correlation Analysis...")
    statistical_results = []
    
    BASE_GREEDY_STRATEGIES = ['greedy_candidate', 'greedy_K_aggregate']
    UNCERT_METHOD = 'uncert_3'

    pbar_stat = tqdm(total=len(BASE_GREEDY_STRATEGIES) * len(HYPER_K_VALUES), desc="Statistical Analysis")

    for base_greedy_strategy in BASE_GREEDY_STRATEGIES:
        for k_value in HYPER_K_VALUES:
            analysis_df = uncertainty_df[uncertainty_df['k_value'] == k_value].copy()
            if analysis_df.empty:
                pbar_stat.update(1)
                continue

            analysis_df['is_correct'] = (analysis_df[base_greedy_strategy] == analysis_df['ground_truth_label']).astype(int)

            correct_scores = analysis_df[analysis_df['is_correct'] == 1][UNCERT_METHOD]
            incorrect_scores = analysis_df[analysis_df['is_correct'] == 0][UNCERT_METHOD]

            t_stat, t_p, u_p, log_reg_coef, log_reg_p, pseudo_r_sq = (None,) * 6
            if len(correct_scores) > 1 and len(incorrect_scores) > 1:
                t_stat, t_p = ttest_ind(incorrect_scores, correct_scores, equal_var=False, nan_policy='omit')
                try:
                    _, u_p = mannwhitneyu(incorrect_scores, correct_scores, alternative='greater')
                except ValueError:
                    u_p = None

            if analysis_df['is_correct'].nunique() > 1:
                Y = analysis_df['is_correct']
                X = sm.add_constant(analysis_df[UNCERT_METHOD])
                try:
                    logit_model = sm.Logit(Y, X).fit(disp=0)
                    log_reg_coef = logit_model.params[UNCERT_METHOD]
                    log_reg_p = logit_model.pvalues[UNCERT_METHOD]
                    pseudo_r_sq = logit_model.prsquared
                except Exception:
                    log_reg_coef, log_reg_p, pseudo_r_sq = (None,) * 3

            statistical_results.append({
                'base_greedy_strategy': base_greedy_strategy,
                'uncert_method': UNCERT_METHOD,
                'k_value': k_value,
                'mean_uncert_correct': correct_scores.mean(),
                'mean_uncert_incorrect': incorrect_scores.mean(),
                't_statistic': t_stat,
                't_p_value': t_p,
                'mann_whitney_u_p_value': u_p,
                'log_reg_coef': log_reg_coef,
                'log_reg_p_value': log_reg_p,
                'pseudo_r_squared': pseudo_r_sq
            })
            pbar_stat.update(1)
    
    pbar_stat.close()
    stat_results_df = pd.DataFrame(statistical_results)

    # Phase 6: Saving Final Evaluation Reports by Strategy
    print("\n--> Phase 6: Saving Final Evaluation Reports by Strategy...")
    detailed_log_path = f"{args.output_base_path}_detailed_predictions_{timestamp}.json"
    with open(detailed_log_path, 'w') as f:
        json.dump(detailed_predictions_log, f, indent=4)
    print(f"\nDetailed prediction log saved to: {detailed_log_path}")

    eval_df = pd.DataFrame(evaluation_results).round(4)
    
    BASE_STRATEGY_FOLDERS = ['greedy_K_aggregate', 'greedy_candidate']

    for base_folder_name in BASE_STRATEGY_FOLDERS:
        output_dir = os.path.join(args.output_base_path, f"{base_folder_name}_results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Generating reports for base strategy: {base_folder_name} ---")
        
        base_strategy_df = eval_df[eval_df['base_greedy_strategy'] == base_folder_name]

        for cons_strategy in CONSERVATIVE_STRATEGIES:
            strategy_df = base_strategy_df[base_strategy_df['conservative_strategy'] == cons_strategy].copy()
            
            if strategy_df.empty:
                print(f"   - Strategy '{cons_strategy}' generated no results for base '{base_folder_name}', skipping.")
                continue
            
            if 'avg_macro_f1' in strategy_df.columns:
                strategy_df.rename(columns={
                    'avg_macro_f1': 'macro_f1', 'avg_weighted_f1': 'weighted_f1', 'avg_precision': 'precision'
                }, inplace=True)
            
            if cons_strategy == 'NEUTRAL':
                strategy_df = strategy_df.drop(columns=['temperature']).drop_duplicates(
                    subset=['k_value', 'uncert_method', 'threshold_percentile']
                )

            strategy_report_path = os.path.join(output_dir, f"evaluation_report_{base_folder_name}_{cons_strategy}_{timestamp}.csv")
            strategy_df.to_csv(strategy_report_path, index=False)
            print(f"   - Evaluation report for conservative strategy '{cons_strategy}' saved to: {strategy_report_path}")
        
        stat_report_df_filtered = stat_results_df[stat_results_df['base_greedy_strategy'] == base_folder_name]
        
        if not stat_report_df_filtered.empty:
            stat_report_df_to_save = stat_report_df_filtered.drop(columns=['base_greedy_strategy']).copy()
            stat_report_df_to_save.sort_values(by=['uncert_method', 'k_value'], inplace=True)
            stat_report_path = os.path.join(output_dir, f"statistical_correlation_report_{timestamp}.csv")
            stat_report_df_to_save.to_csv(stat_report_path, index=False, float_format='%.4g')
            print(f"   - Statistical correlation report saved to: {stat_report_path}")


    # Phase 7: Reporting Unmapped Tokens
    print("\n--> Phase 7: Reporting Unmapped Tokens for Review...")
    unmapped_set = analysis_helper.unmapped_tokens
    if unmapped_set:
        unmapped_data = []
        for token_id in sorted(list(unmapped_set)):
            decoded_token = tokenizer.decode([token_id])
            unmapped_data.append({'token_id': token_id, 'decoded_token': decoded_token})
        unmapped_df = pd.DataFrame(unmapped_data)
        unmapped_report_path = f"{args.output_base_path}_unmapped_tokens.csv"
        unmapped_df.to_csv(unmapped_report_path, index=False)
        print(f"Found {len(unmapped_set)} token(s) that were predicted but could not be mapped to a canonical label.")
        print(f"A review file has been saved to: {unmapped_report_path}")
    else:
        print("No unmapped tokens were encountered during prediction. All predicted tokens were successfully mapped.")


if __name__ == "__main__":
    main()