#!/usr/bin/env python3
"""
Compute average scores for each model:
- avg(best_of_5): For each problem, take the best score out of 5 runs, then average across all problems
- avg(first_one_shot): For each problem, take the first run's score, then average across all problems

All failed attempts (including timeouts, generation failures) are counted as 0 in the denominator.
"""

import argparse
import csv
import os
import re
from collections import defaultdict

# Mapping-based model normalization
MODEL_MAPPING = {
    "gemini3pro": "gemini3pro",
    "deepseekreasoner": "deepseekreasoner",
    "gpt-4-turbo": "gpt4-turbo",
    "qwen_2.5_72b": "qwen_2.5_72b",
    "gemini2.5pro": "gemini2.5pro",
    "grok4fast": "grok4fastreasoning",
    "gpt5.1": "gpt5.1",
    "gpt5.2": "gpt5.2",
    "gpt5": "gpt5",
    "grok4fastreasoning": "grok4fastreasoning",
    # "claude4.1opus": "claude4.1opus",
    # "claude4.5sonnet": "claude4.5sonnet",
}


def normalize_name(path: str) -> str:
    """Normalize model name via mapping-first.
    Returns None if not in MODEL_MAPPING.
    """
    filename = os.path.basename(path)
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    low = stem.lower()
    sorted_keys = sorted(MODEL_MAPPING.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key.lower() in low:
            return MODEL_MAPPING[key]
    return None


def get_attempt_index(solution_name: str) -> int:
    """
    Extract attempt index from solution name.
    e.g., "0/gpt5.1.cpp" -> 0 (first attempt)
          "0/gpt5.1_1.cpp" -> 1 (second attempt)
          "0/gpt5.1_4.cpp" -> 4 (fifth attempt)
    """
    filename = os.path.basename(solution_name)
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    # Match trailing _N pattern
    match = re.search(r'_(\d+)$', stem)
    if match:
        return int(match.group(1))
    return 0  # First attempt has no suffix


def normalize_problem_name(prob: str, aggregate_subtasks: bool) -> str:
    """
    Normalize problem name. 
    If aggregate_subtasks=True (for research problems):
      - Keep at most the first two path components (separated by /)
      - e.g., "poc_generation/heap_buffer_overflow/arvo_21000" -> "poc_generation/heap_buffer_overflow"
      - e.g., "cant_be_late/high_availability_loose_deadline_large_overhead" -> "cant_be_late/high_availability_loose_deadline_large_overhead"
      - e.g., "cloudcast" -> "cloudcast"
    If aggregate_subtasks=False: return as-is.
    """
    if not aggregate_subtasks:
        return prob
    parts = prob.split('/')
    if len(parts) >= 3:
        # Keep only first two parts
        return '/'.join(parts[:2])
    return prob


def is_valid_problem(prob: str) -> bool:
    """Check if this is a valid problem name (not garbage data)."""
    if not prob or prob.startswith(' ') or 'sky start' in prob or '--retry' in prob:
        return False
    # Skip cant_be_late_multi without subpath (but keep cant_be_late_multi/xxx)
    if prob == 'cant_be_late_multi':
        return False
    return True


def load_results(csv_path, score_field='score', drop_failed=False, drop_timeout=False, aggregate_subtasks=False):
    """
    Load results and organize by (problem, model) -> list of (attempt_index, score)
    
    If drop_failed=False: Failed attempts (Generation failed, timeout, etc.) are treated as score 0.
    If drop_failed=True: All failed attempts are completely dropped (not counted in denominator).
    If drop_timeout=True: Only timeout attempts are dropped; generation failures are treated as score 0.
    If aggregate_subtasks=True: Aggregate subtasks into parent problems by averaging scores 
                                 for the same (agg_problem, model, attempt_idx).
    """
    if aggregate_subtasks:
        # First pass: collect all scores grouped by (agg_problem, model, attempt_idx, orig_problem)
        # Structure: {(agg_prob, model, attempt_idx): [scores from different subtasks]}
        subtask_scores = defaultdict(list)
        
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_prob = row.get('problem', 'unknown')
                
                if not is_valid_problem(orig_prob):
                    continue
                
                agg_prob = normalize_problem_name(orig_prob, aggregate_subtasks=True)
                raw_name = row.get('solution', 'unknown')
                model = normalize_name(raw_name)
                
                if model is None:
                    continue
                
                attempt_idx = get_attempt_index(raw_name)
                message = row.get('message', '')
                status = row.get('status', '')
                
                is_timeout = ('Generation failed' in message or 'Not generated' in message)
                is_generation_failed = (status != 'success')

                if is_timeout:
                    if drop_timeout or drop_failed:
                        continue
                    else:
                        score = 0.0
                elif is_generation_failed:
                    if drop_failed:
                        continue
                    else:
                        score = 0.0
                else:
                    try:
                        score = float(row.get(score_field, 0))
                    except:
                        score = 0.0
                
                key = (agg_prob, model, attempt_idx)
                subtask_scores[key].append(score)
        
        # Second pass: average subtask scores and organize by (problem, model)
        data = defaultdict(lambda: defaultdict(list))
        for (agg_prob, model, attempt_idx), scores in subtask_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            data[agg_prob][model].append((attempt_idx, avg_score))
        
        return data
    
    # Original logic for non-aggregated mode
    # Structure: {problem: {model: [(attempt_index, score), ...]}}
    data = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = row.get('problem', 'unknown')
            raw_name = row.get('solution', 'unknown')
            model = normalize_name(raw_name)
            
            # Skip models not in MODEL_MAPPING
            if model is None:
                continue
            
            attempt_idx = get_attempt_index(raw_name)
            message = row.get('message', '')
            status = row.get('status', '')
            
            # Check failure types
            is_timeout = ('Generation failed' in message or 'Not generated' in message)
            is_generation_failed = (status != 'success')

            # Handle timeout
            if is_timeout:
                if drop_timeout or drop_failed:
                    # Drop timeout attempts
                    continue
                else:
                    score = 0.0
            # Handle generation failure
            elif is_generation_failed:
                if drop_failed:
                    # Drop generation failures
                    continue
                else:
                    score = 0.0
            else:
                try:
                    score = float(row.get(score_field, 0))
                except:
                    score = 0.0

            data[prob][model].append((attempt_idx, min(100, score)))

    return data

def compute_avg_scores(data):
    """
    Compute avg(best_of_5), avg(first_one_shot), and avg(all) for each model.
    """
    # {model: {'best_of_5_scores': [], 'first_shot_scores': [], 'all_scores': []}}
    model_stats = defaultdict(lambda: {'best_of_5_scores': [], 'first_shot_scores': [], 'all_scores': []})
    
    for prob, models_data in data.items():
        for model, attempts in models_data.items():
            # Sort by attempt index
            attempts_sorted = sorted(attempts, key=lambda x: x[0])
            scores = [s for _, s in attempts_sorted]
            
            if len(scores) == 0:
                continue
            
            # Best of 5: max score across all attempts
            best_score = max(scores)
            model_stats[model]['best_of_5_scores'].append(best_score)
            
            # First one-shot: first attempt's score
            first_score = scores[0]
            model_stats[model]['first_shot_scores'].append(first_score)
            
            # All scores: extend with all attempt scores
            model_stats[model]['all_scores'].extend(scores)
    
    # Compute averages
    results = {}
    for model, stats in model_stats.items():
        best_of_5_scores = stats['best_of_5_scores']
        first_shot_scores = stats['first_shot_scores']
        all_scores = stats['all_scores']

        avg_best_of_5 = sum(best_of_5_scores) / len(best_of_5_scores) if best_of_5_scores else 0.0
        avg_first_shot = sum(first_shot_scores) / len(first_shot_scores) if first_shot_scores else 0.0
        avg_all = sum(all_scores) / len(all_scores) if all_scores else 0.0

        results[model] = {
            'avg_best_of_5': avg_best_of_5,
            'avg_first_one_shot': avg_first_shot,
            'avg_all': avg_all,
            'num_problems': len(best_of_5_scores),
            'num_attempts': len(all_scores),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute avg(best_of_5) and avg(first_one_shot) for each model'
    )
    parser.add_argument('csv', help='Input CSV (results.csv)')
    parser.add_argument('--out', default='avg_scores.csv', help='Output CSV path (default: avg_scores.csv)')
    parser.add_argument('--drop-failed', action='store_true', 
                        help='Drop all failed attempts (generation failures + timeouts) instead of counting them as 0')
    parser.add_argument('--drop-timeout', action='store_true',
                        help='Drop only timeout attempts; generation failures are still counted as 0')
    parser.add_argument('--aggregate-subtasks', action='store_true',
                        help='Aggregate subtasks into parent problems (for research problems). '
                             'Problems with 3+ path components (e.g., a/b/c) will be grouped by first 2 components (a/b).')
    args = parser.parse_args()

    data = load_results(args.csv, drop_failed=args.drop_failed, drop_timeout=args.drop_timeout, 
                        aggregate_subtasks=args.aggregate_subtasks)
    results = compute_avg_scores(data)
    
    # Sort by avg_best_of_5 descending
    sorted_res = sorted(results.items(), key=lambda x: -x[1]['avg_best_of_5'])
    
    print(f"\n{'Model':<25} | {'Avg Best-of-5':<15} | {'Avg First-Shot':<15} | {'Avg All':<15} | {'# Problems':<12} | {'# Attempts'}")
    print("-" * 105)
    
    rows = []
    for model, stats in sorted_res:
        avg_b5 = stats['avg_best_of_5']
        avg_fs = stats['avg_first_one_shot']
        avg_all = stats['avg_all']
        n_prob = stats['num_problems']
        n_attempts = stats['num_attempts']
        print(f"{model:<25} | {avg_b5:<15.3f} | {avg_fs:<15.3f} | {avg_all:<15.3f} | {n_prob:<12} | {n_attempts}")
        
        rows.append({
            'model': model,
            'avg_best_of_5': f"{avg_b5:.4f}",
            'avg_first_one_shot': f"{avg_fs:.4f}",
            'avg_all': f"{avg_all:.4f}",
            'num_problems': n_prob,
            'num_attempts': n_attempts,
        })
    
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'avg_best_of_5', 'avg_first_one_shot', 'avg_all', 'num_problems', 'num_attempts'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
