import json
import os
import argparse
import csv


def extract_results(file_path):
    """
    Extracts evaluation results from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the extracted results.
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    with open(file_path, 'r') as f:
        data = json.load(f)

    model_name = data.get("config_general", {}).get("model_name", "N/A")

    results = {}
    total_weighted_acc = 0
    total_docs = 0

    for task_key, task_result in data.get("results", {}).items():
        if "_average" in task_key or task_key == "all":
            continue

        # Create a cleaner task name
        try:
            clean_task_name = task_key.split(":")[1].split("|")[0]
        except IndexError:
            clean_task_name = task_key

        acc_norm = task_result.get("acc_norm")
        if acc_norm is not None:
            results[clean_task_name] = acc_norm

            # For macro accuracy calculation
            task_config_key = task_key.rsplit("|", 1)[0]
            task_config = data.get("config_tasks", {}).get(task_config_key, {})
            num_docs = task_config.get("effective_num_docs")

            if num_docs is not None:
                total_weighted_acc += acc_norm * num_docs
                total_docs += num_docs

    macro_accuracy = total_weighted_acc / total_docs if total_docs > 0 else 0

    return {
        "model_name": model_name,
        "evaluation_results": results,
        "macro_accuracy": macro_accuracy,
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

    args = parser.parse_args()

    json_files = []
    for root, _, files in os.walk(args.folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"No JSON files found in {args.folder_path}")
        exit()

    all_results = []
    all_eval_keys = set()

    for file_path in json_files:
        extracted_data = extract_results(file_path)
        if "error" not in extracted_data:
            file_name = os.path.basename(file_path)
            extracted_data['file_name'] = file_name
            all_results.append(extracted_data)
            all_eval_keys.update(extracted_data.get("evaluation_results", {}).keys())

    sorted_eval_keys = sorted(list(all_eval_keys))
    
    fieldnames = ['file_name', 'model_name', 'macro_accuracy'] + sorted_eval_keys

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {
                'file_name': result.get('file_name'),
                'model_name': result.get('model_name'),
                'macro_accuracy': result.get('macro_accuracy')
            }
            row.update(result.get("evaluation_results", {}))
            writer.writerow(row)
            
    print(f"Combined results saved to {args.output}")
