#!/usr/bin/env python3
"""
DCT Evaluation & Report Generator.

Reads benchmark results and produces:
- Precision, recall, F1
- ROC curves
- Ablation table
- Latency distribution
- Per-category breakdown
- Correlated error analysis

Usage:
    python scripts/evaluate.py --results results/benchmark.jsonl --output results/report/
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute core DCT metrics."""
    total = len(results)
    
    # Baseline: how often does the model miss tool calls?
    should_use_tool = [r for r in results if r["ground_truth_should_use_tool"]]
    should_not_use_tool = [r for r in results if not r["ground_truth_should_use_tool"]]
    
    missed = [r for r in results if r["missed_tool_call"]]
    unnecessary = [r for r in results if r["unnecessary_tool_call"]]
    
    # DCT performance
    true_positives = [r for r in results if r["dct_correct_intervention"]]
    false_positives = [r for r in results if r["dct_false_positive"]]
    false_negatives = [r for r in missed if not r["dct_would_intervene"]]
    
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Latency
    latencies = [r["latency_ms"] for r in results]
    
    return {
        "total_prompts": total,
        "should_use_tool": len(should_use_tool),
        "should_not_use_tool": len(should_not_use_tool),
        "baseline_missed_tool_calls": len(missed),
        "baseline_missed_rate": len(missed) / len(should_use_tool) if should_use_tool else 0,
        "unnecessary_tool_calls": len(unnecessary),
        "dct_true_positives": tp,
        "dct_false_positives": fp,
        "dct_false_negatives": fn,
        "dct_precision": precision,
        "dct_recall": recall,
        "dct_f1": f1,
        "latency_p50": np.percentile(latencies, 50),
        "latency_p95": np.percentile(latencies, 95),
        "latency_p99": np.percentile(latencies, 99),
        "latency_mean": np.mean(latencies),
    }


def per_category_breakdown(results: list[dict]) -> dict:
    """Break down metrics by prompt category."""
    categories = defaultdict(list)
    for r in results:
        categories[r["category"]].append(r)
    
    breakdown = {}
    for cat, cat_results in sorted(categories.items()):
        metrics = compute_metrics(cat_results)
        breakdown[cat] = {
            "count": len(cat_results),
            "missed_rate": metrics["baseline_missed_rate"],
            "dct_recall": metrics["dct_recall"],
            "dct_precision": metrics["dct_precision"],
            "dct_f1": metrics["dct_f1"],
        }
    
    return breakdown


def logprob_distribution_analysis(results: list[dict]) -> dict:
    """Analyze logprob distributions for critical vs non-critical tokens."""
    critical_logprobs = []
    noncritical_logprobs = []
    
    for r in results:
        for t in r.get("token_analysis", []):
            if t["dct_critical"]:
                critical_logprobs.append(t["logprob"])
            else:
                noncritical_logprobs.append(t["logprob"])
    
    return {
        "critical_tokens": {
            "count": len(critical_logprobs),
            "mean_logprob": np.mean(critical_logprobs) if critical_logprobs else 0,
            "std_logprob": np.std(critical_logprobs) if critical_logprobs else 0,
            "p25": np.percentile(critical_logprobs, 25) if critical_logprobs else 0,
            "p50": np.percentile(critical_logprobs, 50) if critical_logprobs else 0,
            "p75": np.percentile(critical_logprobs, 75) if critical_logprobs else 0,
        },
        "noncritical_tokens": {
            "count": len(noncritical_logprobs),
            "mean_logprob": np.mean(noncritical_logprobs) if noncritical_logprobs else 0,
            "std_logprob": np.std(noncritical_logprobs) if noncritical_logprobs else 0,
            "p25": np.percentile(noncritical_logprobs, 25) if noncritical_logprobs else 0,
            "p50": np.percentile(noncritical_logprobs, 50) if noncritical_logprobs else 0,
            "p75": np.percentile(noncritical_logprobs, 75) if noncritical_logprobs else 0,
        },
    }


def correlated_error_analysis(results: list[dict]) -> dict:
    """
    Analyze whether DCT misses correlate with specific patterns.
    
    This checks if DCT's false negatives cluster around specific:
    - Question types
    - Token patterns
    - Logprob ranges
    """
    false_negatives = [
        r for r in results
        if r["missed_tool_call"] and not r["dct_would_intervene"]
    ]
    
    true_positives = [
        r for r in results
        if r["dct_correct_intervention"]
    ]
    
    if not false_negatives:
        return {"message": "No false negatives to analyze"}
    
    # Category distribution of FN vs TP
    fn_categories = defaultdict(int)
    tp_categories = defaultdict(int)
    
    for r in false_negatives:
        fn_categories[r["category"]] += 1
    for r in true_positives:
        tp_categories[r["category"]] += 1
    
    # Logprob analysis of FN: what were the logprobs on critical tokens?
    fn_critical_logprobs = []
    for r in false_negatives:
        for t in r.get("token_analysis", []):
            if t["dct_critical"]:
                fn_critical_logprobs.append(t["logprob"])
    
    return {
        "false_negative_count": len(false_negatives),
        "true_positive_count": len(true_positives),
        "fn_by_category": dict(fn_categories),
        "tp_by_category": dict(tp_categories),
        "fn_critical_token_logprobs": {
            "count": len(fn_critical_logprobs),
            "mean": np.mean(fn_critical_logprobs) if fn_critical_logprobs else None,
            "std": np.std(fn_critical_logprobs) if fn_critical_logprobs else None,
        },
        "analysis": (
            "If FN critical logprobs are HIGH (>-0.3), the model was confident "
            "in its wrong decision -- these are the hardest cases for DCT. "
            "If FN critical logprobs are LOW (<-0.5), the classifier missed them "
            "-- training data needs examples like these."
        ),
    }


def roc_data(results: list[dict]) -> list[dict]:
    """
    Generate ROC curve data points for different logprob thresholds.
    
    Sweep threshold from -0.1 to -2.0 and compute TPR/FPR at each.
    """
    thresholds = np.arange(-0.1, -2.1, -0.1)
    roc_points = []
    
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        for r in results:
            should_intervene = r["missed_tool_call"]
            
            # Would DCT intervene at this threshold?
            would_intervene = False
            for t in r.get("token_analysis", []):
                if t["dct_critical"] and t["logprob"] < threshold:
                    if t.get("epistemic", True):
                        would_intervene = True
                        break
            
            if should_intervene and would_intervene:
                tp += 1
            elif should_intervene and not would_intervene:
                fn += 1
            elif not should_intervene and would_intervene:
                fp += 1
            else:
                tn += 1
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        roc_points.append({
            "threshold": float(threshold),
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    
    return roc_points


def generate_report(results: list[dict], output_dir: Path):
    """Generate full evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = compute_metrics(results)
    categories = per_category_breakdown(results)
    logprob_dist = logprob_distribution_analysis(results)
    correlated = correlated_error_analysis(results)
    roc = roc_data(results)
    
    # Write JSON report
    report = {
        "summary": metrics,
        "per_category": categories,
        "logprob_distribution": logprob_dist,
        "correlated_errors": correlated,
        "roc_curve": roc,
    }
    
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Write human-readable report
    with open(output_dir / "report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DCT EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total prompts:              {metrics['total_prompts']}\n")
        f.write(f"Should use tool:            {metrics['should_use_tool']}\n")
        f.write(f"Should NOT use tool:        {metrics['should_not_use_tool']}\n")
        f.write(f"\n")
        f.write(f"BASELINE (no DCT)\n")
        f.write(f"  Missed tool calls:        {metrics['baseline_missed_tool_calls']} ({metrics['baseline_missed_rate']:.1%})\n")
        f.write(f"  Unnecessary tool calls:   {metrics['unnecessary_tool_calls']}\n")
        f.write(f"\n")
        f.write(f"DCT PERFORMANCE\n")
        f.write(f"  True positives:           {metrics['dct_true_positives']}\n")
        f.write(f"  False positives:          {metrics['dct_false_positives']}\n")
        f.write(f"  False negatives:          {metrics['dct_false_negatives']}\n")
        f.write(f"  Precision:                {metrics['dct_precision']:.3f}\n")
        f.write(f"  Recall:                   {metrics['dct_recall']:.3f}\n")
        f.write(f"  F1:                       {metrics['dct_f1']:.3f}\n")
        f.write(f"\n")
        f.write(f"LATENCY\n")
        f.write(f"  P50:                      {metrics['latency_p50']:.0f}ms\n")
        f.write(f"  P95:                      {metrics['latency_p95']:.0f}ms\n")
        f.write(f"  P99:                      {metrics['latency_p99']:.0f}ms\n")
        f.write(f"  Mean:                     {metrics['latency_mean']:.0f}ms\n")
        
        f.write(f"\nPER-CATEGORY BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Category':15s} {'Count':>6s} {'Missed%':>8s} {'Recall':>8s} {'Precision':>10s} {'F1':>6s}\n")
        f.write("-" * 70 + "\n")
        for cat, m in categories.items():
            f.write(f"{cat:15s} {m['count']:>6d} {m['missed_rate']:>7.1%} {m['dct_recall']:>7.3f} {m['dct_precision']:>9.3f} {m['dct_f1']:>6.3f}\n")
        
        f.write(f"\nROC CURVE DATA (threshold sweep)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Threshold':>10s} {'TPR':>8s} {'FPR':>8s} {'Precision':>10s} {'TP':>5s} {'FP':>5s} {'FN':>5s}\n")
        f.write("-" * 70 + "\n")
        for pt in roc:
            f.write(f"{pt['threshold']:>+10.1f} {pt['tpr']:>7.3f} {pt['fpr']:>7.3f} {pt['precision']:>9.3f} {pt['tp']:>5d} {pt['fp']:>5d} {pt['fn']:>5d}\n")
        
        f.write(f"\nCORRELATED ERROR ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(json.dumps(correlated, indent=2, default=str))
        f.write("\n")
    
    print(f"Report written to {output_dir}")
    print(f"  report.json  - machine-readable")
    print(f"  report.txt   - human-readable")
    
    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"DCT Precision: {metrics['dct_precision']:.3f}")
    print(f"DCT Recall:    {metrics['dct_recall']:.3f}")
    print(f"DCT F1:        {metrics['dct_f1']:.3f}")
    print(f"Baseline miss: {metrics['baseline_missed_rate']:.1%}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DCT benchmark results")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/report/")
    args = parser.parse_args()
    
    results = load_results(args.results)
    print(f"Loaded {len(results)} results from {args.results}")
    
    generate_report(results, Path(args.output))


if __name__ == "__main__":
    main()
