#!/usr/bin/env python3
"""
Pairwise Soft-Outcome BT with Confidence Intervals
- Calculates Elo based on "Stability/Random" assumption.
- Computes standard errors using the Hessian (Fisher Information) of the MAP estimate.
- Output: Rating +/- 95% Confidence Interval.
"""

import argparse
import csv
import math
import os
import re
import sys
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.linalg import inv, pinv

# --- Configuration ---
INITIAL_RATING = 1200.0
PRIOR_SIGMA = 60.0
BETA = math.log(10) / 400.0

# Mapping-based model normalization
MODEL_MAPPING = {
    # Examples provided by user
    "gemini3pro": "gemini3pro",
    "deepseekreasoner": "deepseekreasoner",
    "gpt-4-turbo": "gpt4-turbo",
    "claude-3-opus": "claude3-opus",
    "qwen_2.5_72b": "qwen_2.5_72b",
    "gemini2.5pro": "gemini2.5pro",
    "grok4fast": "grok4fastreasoning",
    # Dataset-specific common names
    "gpt5.1": "gpt5.1",
    "gpt5.2": "gpt5.2",
    "gpt5": "gpt5",
    "grok4fastreasoning": "grok4fastreasoning",
    # "claude4.1opus": "claude4.1opus",
    # "claude4.5sonnet": "claude4.5sonnet",
}

def normalize_name(path: str) -> str:
    """Normalize model name via mapping-first.

    1) Take basename and strip final extension
    2) If any mapping key is a substring of the stem, return its mapped value
    3) If not in MODEL_MAPPING, return None (will be filtered out)
    """
    filename = os.path.basename(path)
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    low = stem.lower()
    if "low" in low or "medium" in low or "high" in low:
        return None
    # Match longer keys first to avoid substring collisions (e.g., gpt5.1 vs gpt5)
    sorted_keys = sorted(MODEL_MAPPING.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key.lower() in low:
            return MODEL_MAPPING[key]
    # Not in MODEL_MAPPING, return None to filter out
    return None

def normalize_problem_name(prob: str, aggregate_subtasks: bool) -> str:
    """
    Normalize problem name for aggregation.
    If aggregate_subtasks=True: Keep at most first two path components.
    """
    if not aggregate_subtasks:
        return prob
    parts = prob.split('/')
    if len(parts) >= 3:
        return '/'.join(parts[:2])
    return prob


def is_valid_problem(prob: str) -> bool:
    """Check if this is a valid problem name (not garbage data)."""
    if not prob or prob.startswith(' ') or 'sky start' in prob or '--retry' in prob:
        return False
    if prob == 'cant_be_late_multi':
        return False
    return True


def get_attempt_index(solution_name: str) -> int:
    """Extract attempt index from solution name."""
    filename = os.path.basename(solution_name)
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    match = re.search(r'_(\d+)$', stem)
    if match:
        return int(match.group(1))
    return 0


def calculate_expected_outcome(scores_a, scores_b):
    total_score = 0.0
    count = 0
    for sa in scores_a:
        for sb in scores_b:
            if sa > sb + 1e-9: total_score += 1.0
            elif sb > sa + 1e-9: total_score += 0.0
            else: total_score += 0.5
            count += 1
    return total_score / count if count > 0 else 0.5

def load_and_build_soft_pairs(csv_path, score_field='score_unbounded', aggregate_subtasks=False):
    """
    Load results and build soft pairwise comparisons.
    
    If aggregate_subtasks=True: 
      - Aggregate subtasks by averaging scores for same (agg_problem, model, attempt_idx)
      - Then treat aggregated problems as single problems for ELO calculation
      - Uses score_field (default: score_unbounded) with fallback to 'score'
    """
    if aggregate_subtasks:
        # First pass: collect scores grouped by (agg_prob, model, attempt_idx)
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
                
                message = row.get('message', '')
                
                attempt_idx = get_attempt_index(raw_name)
                
                # Use score_field first, fall back to score
                score_str = row.get(score_field, '')
                if not score_str:
                    score_str = row.get('score', '')
                try:
                    s = float(score_str) if score_str else 0.0
                except:
                    s = 0.0
                subtask_scores[(agg_prob, model, attempt_idx)].append(s)
        
        # Second pass: average subtask scores and organize by problem
        by_problem = defaultdict(list)
        for (agg_prob, model, attempt_idx), scores in subtask_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            by_problem[agg_prob].append((model, avg_score))
    else:
        # Original logic
        by_problem = defaultdict(list)
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prob = row.get('problem', 'unknown')
                raw_name = row.get('solution', 'unknown')
                model = normalize_name(raw_name)
                if model is None:
                    continue
                message = row.get('message', '')
                # Try score_field first, fall back to score
                score_val = row.get(score_field, '')
                if not score_val:
                    score_val = row.get('score', '')
                try:
                    s = float(score_val) if score_val else 0.0
                except:
                    s = 0.0
                by_problem[prob].append((model, s))

    matches = []
    all_models = set()
    print(f"Processing {len(by_problem)} problems using Average-Linkage...")
    for prob, items in by_problem.items():
        model_scores = defaultdict(list)
        for m, s in items: model_scores[m].append(s)
        models = list(model_scores.keys())
        all_models.update(models)
        n = len(models)
        if n < 2: continue
        for i in range(n):
            for j in range(i + 1, n):
                mA, mB = models[i], models[j]
                outcome = calculate_expected_outcome(model_scores[mA], model_scores[mB])
                matches.append((mA, mB, outcome))
    return matches, sorted(list(all_models))

def compute_hessian(r, idx_i, idx_j, games, reg_coeff):
    """
    Computes the Hessian matrix (2nd derivative) at the optimal ratings.
    H_ii = Prior + Sum( matches * P * (1-P) * beta^2 )
    H_ij = - matches * P * (1-P) * beta^2
    """
    n = len(r)
    H = np.zeros((n, n))
    
    # 1. Add Prior Curvature (Diagonal)
    np.fill_diagonal(H, reg_coeff)
    
    # 2. Add Likelihood Curvature
    r_i = r[idx_i]
    r_j = r[idx_j]
    diff = (r_i - r_j) * BETA
    p_vals = 1.0 / (1.0 + np.exp(-diff))
    
    # Second derivative coefficient for logistic loss
    weights = games * p_vals * (1.0 - p_vals) * (BETA ** 2)
    
    # Accumulate into Hessian
    np.add.at(H, (idx_i, idx_i), weights)
    np.add.at(H, (idx_j, idx_j), weights)
    np.add.at(H, (idx_i, idx_j), -weights)
    np.add.at(H, (idx_j, idx_i), -weights)
    
    return H

def optimize_bt_with_ci(matches, models):
    model_map = {m: i for i, m in enumerate(models)}
    n_models = len(models)
    
    pair_stats = defaultdict(lambda: {'wins': 0.0, 'games': 0.0})
    for mA, mB, score in matches:
        i, j = model_map[mA], model_map[mB]
        if i > j: i, j, score = j, i, 1.0 - score
        pair_stats[(i, j)]['wins'] += score
        pair_stats[(i, j)]['games'] += 1.0
        
    idx_i, idx_j, wins_i, games = [], [], [], []
    for (i, j), stat in pair_stats.items():
        idx_i.append(i); idx_j.append(j)
        wins_i.append(stat['wins']); games.append(stat['games'])
    
    idx_i = np.array(idx_i, dtype=np.int32)
    idx_j = np.array(idx_j, dtype=np.int32)
    wins_i = np.array(wins_i, dtype=np.float64)
    games = np.array(games, dtype=np.float64)
    
    reg_coeff = 1.0 / (PRIOR_SIGMA ** 2)

    def objective(r):
        dev = r - INITIAL_RATING
        loss = 0.5 * reg_coeff * np.sum(dev**2)
        grad = reg_coeff * dev
        
        r_i, r_j = r[idx_i], r[idx_j]
        diff = (r_i - r_j) * BETA
        log_1p_exp_neg = np.logaddexp(0, -diff)
        p_vals = 1.0 / (1.0 + np.exp(-diff))
        
        losses_i = games - wins_i
        batch_loss = wins_i * log_1p_exp_neg + losses_i * (diff + log_1p_exp_neg)
        loss += np.sum(batch_loss)
        
        err = (p_vals * games) - wins_i
        g_diff = err * BETA
        np.add.at(grad, idx_i, g_diff)
        np.add.at(grad, idx_j, -g_diff)
        return loss, grad

    # 1. Optimize
    res = minimize(objective, np.full(n_models, INITIAL_RATING),
                   method='L-BFGS-B', jac=True, options={'ftol': 1e-9})
    r_opt = res.x
    
    # 2. Compute Uncertainty (Hessian)
    H = compute_hessian(r_opt, idx_i, idx_j, games, reg_coeff)
    
    try:
        cov = inv(H)
    except np.linalg.LinAlgError:
        cov = pinv(H)
        
    std_errs = np.sqrt(np.diag(cov))
    
    results = {}
    for i, m in enumerate(models):
        results[m] = {'rating': r_opt[i], 'se': std_errs[i]}
    
    return results

def main():
    csv.field_size_limit(sys.maxsize)
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Input CSV')
    parser.add_argument('--out', default='elo_ratings.csv', help='Output CSV path (default: elo_ratings.csv)')
    parser.add_argument('--aggregate-subtasks', action='store_true',
                        help='Aggregate subtasks into parent problems (for research problems). '
                             'Problems with 3+ path components will be grouped by first 2 components, '
                             'with scores averaged per (problem, model, attempt).')
    args = parser.parse_args()

    matches, models = load_and_build_soft_pairs(args.csv, aggregate_subtasks=args.aggregate_subtasks)
    results = optimize_bt_with_ci(matches, models)

    # Center ratings
    vals = [d['rating'] for d in results.values()]
    shift = 1200.0 - np.mean(vals)
    
    sorted_res = sorted(results.items(), key=lambda x: -x[1]['rating'])
    
    print(f"{'Model':<25} | {'Rating':<8} | {'95% CI':<15} | {'Data Points'}")
    print("-" * 65)
    
    rows = []
    for m, d in sorted_res:
        r = d['rating'] + shift
        se = d['se']
        ci = 1.96 * se
        print(f"{m:<25} | {r:<8.1f} | ± {ci:<5.1f}         | (SE={se:.1f})")
        
        rows.append({
            'model': m,
            'rating': f"{r:.2f}",
            'lower_ci_95': f"{r - ci:.2f}",
            'upper_ci_95': f"{r + ci:.2f}",
            'std_err': f"{se:.2f}"
        })

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'rating', 'lower_ci_95', 'upper_ci_95', 'std_err'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved detailed results with CIs to {args.out}")

if __name__ == "__main__":
    main()
