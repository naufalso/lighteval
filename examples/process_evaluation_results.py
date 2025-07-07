import os
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import HfFileSystem, HfApi
from tqdm import tqdm


def setup_environment(hf_home: str = None):
    """Setup the environment variables for Hugging Face."""
    if hf_home:
        os.environ["HF_HOME"] = hf_home


def get_dataset_subsets(repo_path: str) -> List[str]:
    """Get all dataset subsets from the HuggingFace repository."""
    fs = HfFileSystem()
    return [
        data['name'].split('/')[-1] 
        for data in fs.ls(repo_path) 
        if data['type'] == 'directory' and data['name'] != "evaluation_results"
    ]


def get_all_correct_ids(dataset_path: str, data_files_pattern: str) -> Tuple[any, List[str], Dict[str, float]]:
    """Get all IDs with correct answers and their ratios."""
    correct_answer_ratio = defaultdict(int)
    all_correct_ids = []

    dataset = load_dataset(dataset_path, data_files=data_files_pattern + "/*.parquet", split="train")
    dataset = dataset.flatten()
    dataset_pd = dataset.to_pandas()
    grouped = dataset_pd.groupby('specifics.id')

    for group_id, group_df in tqdm(grouped, desc="Processing groups", total=len(grouped)):
        total_correct_answer = group_df['metrics.acc_norm'].sum()
        total_questions = len(group_df)
        correct_answer_ratio[group_id] = total_correct_answer / total_questions if total_questions > 0 else 0

        if total_correct_answer >= total_questions:
            all_correct_ids.append(group_id)

    return dataset, all_correct_ids, correct_answer_ratio


def calculate_models_accuracy(dataset_flatten: any, filtered_ids: List[str] = None) -> Dict[str, float]:
    """Calculate accuracy scores for all models."""
    scores = defaultdict(float)
    dataset = dataset_flatten

    if filtered_ids is not None:
        print(f"Total Question Before Filter: {len(dataset_flatten)}")
        dataset = dataset_flatten.filter(lambda x: x['specifics.id'] not in filtered_ids)
        print(f"Total Question After Filter: {len(dataset)}")

    dataset_pd = dataset.to_pandas()
    grouped = dataset_pd.groupby('model_name')
    
    for group_id, group_df in tqdm(grouped, desc="Processing groups", total=len(grouped)):
        average_score = group_df['metrics.acc_norm'].mean()
        scores[group_id] = average_score

    return scores


def compute_macro_micro_averages(evaluation_results: Dict, dataset_totals: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute macro and micro averages for all models."""
    macro_averages = {}
    micro_averages = {}

    for model_name in evaluation_results[next(iter(evaluation_results))].keys():
        total_score = 0
        total_count = 0

        for subset, scores in evaluation_results.items():
            if model_name in scores:
                score = scores[model_name]
                count = dataset_totals[subset]
                total_score += score * count
                total_count += count

        if total_count > 0:
            macro_averages[model_name] = total_score / total_count
            micro_averages[model_name] = sum(scores[model_name] for scores in evaluation_results.values()) / len(evaluation_results)

    return macro_averages, micro_averages


def save_results(output_dir: str, results: Dict):
    """Save evaluation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    for filename, data in results.items():
        filepath = os.path.join(output_dir, f"{filename}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)


def upload_to_hub(folder_path: str, repo_id: str, path_in_repo: str = "evaluation_results"):
    """Upload results to Hugging Face Hub."""
    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload evaluation results for CyberSec MCQA",
    )


def main(args):
    # Setup environment
    setup_environment(args.hf_home)

    # Get dataset subsets
    subsets = get_dataset_subsets(args.repo_path)
    print(f"Found {len(subsets)} subsets: {subsets}")

    # Initialize result containers
    all_evaluation_results = {}
    all_evaluation_results_filtered = {}
    all_correct_ids = {}
    all_correct_ratios = {}
    dataset_total = defaultdict(int)
    dataset_total_filtered = defaultdict(int)

    # Process each subset
    for subset in subsets:
        print(f"Processing subset: {subset}")
        
        # Get all correct IDs and ratios
        dataset, all_correct_ids[subset], all_correct_ratios[subset] = get_all_correct_ids(
            args.dataset_path, subset
        )
        
        # Calculate model scores
        all_evaluation_results[subset] = calculate_models_accuracy(dataset)
        all_evaluation_results_filtered[subset] = calculate_models_accuracy(
            dataset,
            filtered_ids=all_correct_ids[subset]
        )
        
        dataset_total[subset] = len(all_correct_ratios[subset])
        dataset_total_filtered[subset] = len(all_correct_ids[subset])

    # Compute averages
    macro_avg, micro_avg = compute_macro_micro_averages(all_evaluation_results, dataset_total)
    macro_avg_filtered, micro_avg_filtered = compute_macro_micro_averages(all_evaluation_results_filtered, dataset_total_filtered)

    # Add averages to results
    all_evaluation_results.update({
        "macro_averages": macro_avg,
        "micro_averages": micro_avg
    })
    all_evaluation_results_filtered.update({
        "macro_averages": macro_avg_filtered,
        "micro_averages": micro_avg_filtered
    })

    # Save results
    results = {
        "all_evaluation_results": all_evaluation_results,
        "all_evaluation_results_filtered": all_evaluation_results_filtered,
        "all_correct_ids": all_correct_ids,
        "all_correct_ratios": all_correct_ratios,
        "dataset_total": dataset_total,
        "dataset_total_filtered": dataset_total_filtered
    }
    
    save_results(args.output_dir, results)
    print(f"Results saved to {args.output_dir}")

    # Upload to HuggingFace Hub if specified
    if args.upload_to_hub:
        upload_to_hub(args.output_dir, args.repo_id)
        print(f"Results uploaded to {args.repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MCQA evaluation results")
    parser.add_argument("--hf_home", type=str, help="Path to HuggingFace home directory")
    parser.add_argument("--repo_path", type=str, default="datasets/naufalso/cybersec_mcqa_results/",
                      help="Path to the HuggingFace repository")
    parser.add_argument("--dataset_path", type=str, default="naufalso/cybersec_mcqa_results",
                      help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                      help="Directory to save the results")
    parser.add_argument("--upload_to_hub", action="store_true",
                      help="Whether to upload results to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str, default="naufalso/cybersec_mcqa_results",
                      help="HuggingFace repository ID for uploading results")

    args = parser.parse_args()
    main(args)