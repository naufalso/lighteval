"""
Lighteval Results Extraction Tool

This script extracts evaluation results from lighteval JSON output files and combines them into a CSV.
It supports multiple metrics beyond just 'acc_norm', including accuracy, f1_score, rouge, bleu, and many others.

Key Features:
- Auto-detects available metrics in JSON fi            writer.writerow(filtered_row)
            
    print(f"Combined results saved to {args.output}")
    print(f"Found {len(sorted_metrics)} unique metrics: {', '.join(sorted_metrics)}")
    print(f"Found {len(sorted_eval_keys)} unique tasks: {', '.join(sorted_eval_keys)}")
    print(f"Output CSV contains {len(filtered_fieldnames)} columns (empty columns removed)") Supports multiple metrics per task (not just acc_norm)
- Calculates macro scores using the primary metric
- Supports filtering by folder names
- Can include or exclude standard error metrics
- Provides a list-metrics mode for exploration

Usage Examples:
  # Basic usage with auto-detected primary metric
  python extract_evaluation_results.py results/ -o combined.csv

  # Use specific primary metric
  python extract_evaluation_results.py results/ -m accuracy -o combined.csv

  # Include standard error metrics
  python extract_evaluation_results.py results/ --include-stderr -o combined.csv

  # List available metrics without generating CSV
  python extract_evaluation_results.py results/ --list-metrics

  # Filter by folder pattern
  python extract_evaluation_results.py results/ --filter qwen -o qwen_results.csv

  # Output only metrics without macro calculation
  python extract_evaluation_results.py results/ --metrics-only -o metrics_only.csv

Supported Metrics:
The script automatically detects and supports any metric found in the JSON files, including but not limited to:
- acc_norm (normalized accuracy)
- accuracy
- f1_score, f1_score_macro, f1_score_micro
- exact_match, quasi_exact_match
- rouge, rouge1, rouge2, rougeL
- bleu, bleu_1, bleu_4
- loglikelihood_acc, loglikelihood_f1
- mcc (Matthews correlation coefficient)
- mrr (Mean reciprocal rank)
- And many others...
"""

import json
import os
import argparse
import csv


def extract_results(file_path, primary_metric=None, include_stderr=False):
    """
    Extracts evaluation results from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        primary_metric (str): The primary metric to extract for macro calculation. 
                             If None, auto-detects the most common metric.
        include_stderr (bool): Whether to include standard error metrics.

    Returns:
        dict: A dictionary containing the extracted results.
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    with open(file_path, 'r') as f:
        data = json.load(f)

    model_name = data.get("config_general", {}).get("model_name", "N/A")

    results = {}
    metric_counts = {}
    total_weighted_score = 0
    total_docs = 0

    for task_key, task_result in data.get("results", {}).items():
        if "_average" in task_key or task_key == "all":
            continue

        # Create a cleaner task name
        try:
            clean_task_name = task_key.split("|")[1] #.split("|")[0]
        except IndexError:
            clean_task_name = task_key

        # Extract all metrics (optionally excluding _stderr variants)
        task_metrics = {}
        for metric_name, metric_value in task_result.items():
            # Include metric if it's not stderr or if stderr is explicitly requested
            include_metric = True
            if metric_name.endswith("_stderr") and not include_stderr:
                include_metric = False
            
            if include_metric and isinstance(metric_value, (int, float)):
                task_metrics[metric_name] = metric_value
                # Count metric usage for auto-detection (exclude stderr from counting)
                if not metric_name.endswith("_stderr"):
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

        if task_metrics:
            results[clean_task_name] = task_metrics

    # Determine primary metric for macro calculation
    if not primary_metric and metric_counts:
        primary_metric = max(metric_counts, key=metric_counts.get)

    # Calculate macro score using the primary metric
    macro_score = 0
    if primary_metric:
        weighted_total = 0
        total_docs = 0
        unweighted_total = 0
        unweighted_count = 0
        
        for task_key, task_result in data.get("results", {}).items():
            if "_average" in task_key or task_key == "all":
                continue

            metric_value = task_result.get(primary_metric)
            if metric_value is not None:
                # Try to get weighted calculation first
                task_config_key = task_key.rsplit("|", 1)[0]
                task_config = data.get("config_tasks", {}).get(task_config_key, {})
                num_docs = task_config.get("effective_num_docs")

                if num_docs is not None:
                    weighted_total += metric_value * num_docs
                    total_docs += num_docs
                else:
                    # Fallback to unweighted average
                    unweighted_total += metric_value
                    unweighted_count += 1
        
        # Use weighted average if available, otherwise use unweighted
        if total_docs > 0:
            macro_score = weighted_total / total_docs
        elif unweighted_count > 0:
            macro_score = unweighted_total / unweighted_count

    return {
        "model_name": model_name.replace("models_", "").replace("_", "/"),
        "evaluation_results": results,
        "primary_metric": primary_metric,
        f"macro_{primary_metric}": macro_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract evaluation results from lighteval JSON output files in a folder and combine them into a CSV."
    )
    parser.add_argument("folder_path", type=str, help="Path to the folder containing JSON results files.")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="combined_results.csv",
        help="Path to save the combined results in a CSV file.",
    )

    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        help="Filter pattern to match folder names (e.g., 'qwen' for folders containing 'qwen')",
    )

    parser.add_argument(
        "-m",
        "--primary-metric",
        type=str,
        help="Primary metric to use for macro calculation (e.g., 'acc_norm', 'accuracy'). If not provided, auto-detects the most common metric.",
    )

    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only output metrics columns (no macro calculation).",
    )

    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics found in the JSON files without generating CSV output.",
    )

    parser.add_argument(
        "--include-stderr",
        action="store_true",
        help="Include standard error metrics (e.g., acc_norm_stderr) in the output.",
    )

    args = parser.parse_args()

    json_files = []
    for root, _, files in os.walk(args.folder_path):
        # Apply folder filter if specified
        if args.filter and args.filter.lower() not in root.lower():
            continue
        
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    if not json_files:
        filter_msg = f" with filter pattern '{args.filter}'" if args.filter else ""
        print(f"No JSON files found in {args.folder_path}{filter_msg}")
        exit()

    all_results = []
    all_eval_keys = set()
    all_metrics = set()

    for file_path in json_files:
        extracted_data = extract_results(file_path, primary_metric=args.primary_metric, include_stderr=args.include_stderr)
        if "error" not in extracted_data:
            file_name = os.path.basename(file_path)
            extracted_data['file_name'] = file_name
            all_results.append(extracted_data)
            
            # Collect all evaluation task names and metric names
            eval_results = extracted_data.get("evaluation_results", {})
            all_eval_keys.update(eval_results.keys())
            
            # Collect all unique metrics across all tasks
            for task_metrics in eval_results.values():
                if isinstance(task_metrics, dict):
                    all_metrics.update(task_metrics.keys())

    if not all_results:
        print("No valid results found in the JSON files.")
        exit()

    # Handle --list-metrics option
    if args.list_metrics:
        sorted_metrics = sorted(list(all_metrics))
        print(f"Found {len(sorted_metrics)} unique metrics:")
        for metric in sorted_metrics:
            print(f"  - {metric}")
        print(f"\nFound {len(all_eval_keys)} unique tasks:")
        for task in sorted(all_eval_keys):
            print(f"  - {task}")
        exit()

    # Determine which metrics to include in the output
    sorted_eval_keys = sorted(list(all_eval_keys))
    sorted_metrics = sorted(list(all_metrics))
    
    # Create column structure: basic info + macro score + task-specific metrics
    basic_columns = ['file_name', 'model_name']
    
    if not args.metrics_only and all_results:
        primary_metric = all_results[0].get('primary_metric', 'score')
        macro_column = f'macro_{primary_metric}'
        basic_columns.append(macro_column)
    
    # Create columns for each task-metric combination
    task_metric_columns = []
    for task_name in sorted_eval_keys:
        for metric_name in sorted_metrics:
            task_metric_columns.append(f"{task_name}_{metric_name}")
    
    fieldnames = basic_columns + task_metric_columns

    # First pass: collect all rows to identify non-empty columns
    rows_data = []
    columns_with_data = set(basic_columns)  # Always keep basic columns
    
    for result in all_results:
        row = {
            'file_name': result.get('file_name'),
            'model_name': result.get('model_name'),
        }
        
        # Add macro score if not metrics-only mode
        if not args.metrics_only:
            primary_metric = result.get('primary_metric', 'score')
            macro_column = f'macro_{primary_metric}'
            macro_value = result.get(macro_column)
            row[macro_column] = macro_value
            if macro_value is not None and macro_value != '':
                columns_with_data.add(macro_column)
        
        # Add task-specific metrics
        eval_results = result.get("evaluation_results", {})
        for task_name, task_metrics in eval_results.items():
            if isinstance(task_metrics, dict):
                for metric_name, metric_value in task_metrics.items():
                    column_name = f"{task_name}_{metric_name}"
                    if column_name in fieldnames:
                        row[column_name] = metric_value
                        # Track columns that have actual data
                        if metric_value is not None and metric_value != '':
                            columns_with_data.add(column_name)
            
        rows_data.append(row)
    
    # Filter fieldnames to only include columns with data
    filtered_fieldnames = [col for col in fieldnames if col in columns_with_data]
    
    # Write CSV with only non-empty columns
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=filtered_fieldnames)
        writer.writeheader()
        
        for row in rows_data:
            # Filter row to only include columns that will be in the output
            filtered_row = {col: row.get(col) for col in filtered_fieldnames}
            writer.writerow(filtered_row)
            
    print(f"Combined results saved to {args.output}")
    print(f"Found {len(sorted_metrics)} unique metrics: {', '.join(sorted_metrics)}")
    print(f"Found {len(sorted_eval_keys)} unique tasks: {', '.join(sorted_eval_keys)}")
