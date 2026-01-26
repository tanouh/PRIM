#### Evaluate model performance: precision, accuracy, recall, and execution time
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pandas as pd

# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Evaluate model scores with metrics")
    p.add_argument("--scores_csv", required=True, help="CSV file with query and average scores")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for positive prediction")
    p.add_argument("--out_dir", required=True, help="Output directory for results")
    p.add_argument("--distance_type", choices=["cosine", "euclidean"], default="cosine",
                   help="Distance metric used (affects interpretation)")
    
    return p.parse_args()

# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------
def evaluate_scores(scores_df, threshold, distance_type="cosine"):
    """
    Evaluate model performance based on scores and threshold.
    
    For threshold interpretation:
    - cosine distance: lower scores = more similar; prediction is True if score < threshold
    - euclidean distance: lower scores = more similar; prediction is True if score < threshold
    
    Ground truth is True if query and gallery have the same label.
    """
    start_time = time.time()
    
    # Extract data
    query_labels = scores_df['query_label'].values
    avg_scores = scores_df['avg_score'].values
    
    # Extract prediction time if available
    prediction_time = scores_df['prediction_time_seconds'].iloc[0] if 'prediction_time_seconds' in scores_df.columns else None
    
    # Ground truth: All queries should match their corresponding gallery (same label)
    # Since we're comparing each query against its own label's gallery
    ground_truth = np.ones(len(query_labels), dtype=int)
    
    # Predictions based on threshold
    # Lower score = more similar, so predict match (1) if score < threshold
    predictions = (avg_scores < threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    
    evaluation_time = time.time() - start_time
    
    # Compute additional statistics
    min_score = avg_scores.min()
    max_score = avg_scores.max()
    mean_score = avg_scores.mean()
    std_score = avg_scores.std()
    median_score = np.median(avg_scores)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold": float(threshold),
        "distance_type": distance_type,
        "num_samples": int(len(query_labels)),
        "num_positive_predictions": int(predictions.sum()),
        "num_negative_predictions": int((1 - predictions).sum()),
        "prediction_time_seconds": float(prediction_time) if prediction_time is not None else None,
        "evaluation_time_seconds": float(evaluation_time),
        "score_statistics": {
            "min": float(min_score),
            "max": float(max_score),
            "mean": float(mean_score),
            "std": float(std_score),
            "median": float(median_score),
        }
    }
    
    return metrics, predictions, avg_scores

# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def main():
    args = parse_args()
    
    # Create output directory if needed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scores
    scores_df = pd.read_csv(args.scores_csv)
    
    print(f"Loaded {len(scores_df)} predictions from {args.scores_csv}")
    
    # Evaluate
    metrics, predictions, scores = evaluate_scores(
        scores_df, 
        args.threshold, 
        args.distance_type
    )
    
    # Save metrics to JSON
    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Samples:   {metrics['num_samples']}")
    if metrics['prediction_time_seconds'] is not None:
        print(f"Prediction Time: {metrics['prediction_time_seconds']:.4f}s")
    print(f"Evaluation Time: {metrics['evaluation_time_seconds']:.4f}s")
    print(f"\n=== Score Statistics ===")
    print(f"Min:    {metrics['score_statistics']['min']:.6f}")
    print(f"Max:    {metrics['score_statistics']['max']:.6f}")
    print(f"Mean:   {metrics['score_statistics']['mean']:.6f}")
    print(f"Std:    {metrics['score_statistics']['std']:.6f}")
    print(f"Median: {metrics['score_statistics']['median']:.6f}")
    
    # Save detailed results
    results_df = scores_df.copy()
    results_df['prediction'] = predictions
    results_df['ground_truth'] = 1  # All should match
    results_df['correct'] = (results_df['prediction'] == results_df['ground_truth']).astype(int)
    
    results_csv_path = out_dir / "evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    # Save summary to text file
    summary_path = out_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Model Evaluation Summary ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Threshold: {metrics['threshold']}\n")
        f.write(f"  Distance Type: {metrics['distance_type']}\n")
        f.write(f"  Total Samples: {metrics['num_samples']}\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
        
        f.write(f"Predictions:\n")
        f.write(f"  Positive: {metrics['num_positive_predictions']}\n")
        f.write(f"  Negative: {metrics['num_negative_predictions']}\n\n")
        
        f.write(f"Execution Times:\n")
        if metrics['prediction_time_seconds'] is not None:
            f.write(f"  Prediction: {metrics['prediction_time_seconds']:.4f}s\n")
        f.write(f"  Evaluation: {metrics['evaluation_time_seconds']:.4f}s\n\n")
        
        f.write(f"Score Statistics:\n")
        f.write(f"  Min:    {metrics['score_statistics']['min']:.6f}\n")
        f.write(f"  Max:    {metrics['score_statistics']['max']:.6f}\n")
        f.write(f"  Mean:   {metrics['score_statistics']['mean']:.6f}\n")
        f.write(f"  Std:    {metrics['score_statistics']['std']:.6f}\n")
        f.write(f"  Median: {metrics['score_statistics']['median']:.6f}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {metrics_json_path}")
    print(f"  - {results_csv_path}")
    print(f"  - {summary_path}")

if __name__ == "__main__":
    main()
