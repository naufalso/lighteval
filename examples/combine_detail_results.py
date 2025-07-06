import os
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict

def combine_results_memory_efficient(folder_path, output_dir):
    """
    Combines evaluation results in a memory-efficient way. It first scans all files,
    then processes and saves results for one evaluation type at a time.
    Handles both flat and nested model directory structures.

    The expected folder structures are:
    1. [folder_path]/[model_name]/[evaluation_time]/[results].parquet
    2. [folder_path]/[model_org]/[model_name]/[evaluation_time]/[results].parquet

    Args:
        folder_path (str): The path to the root folder containing the results.
        output_dir (str): The directory to save the combined parquet files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. First pass: Scan for files and group them by evaluation name without loading data.
    files_by_eval = defaultdict(list)
    print("Scanning for files...")
    for entry in tqdm(os.listdir(folder_path), desc="Scanning models"):
        path1 = os.path.join(folder_path, entry)
        if not os.path.isdir(path1):
            continue

        # Check if path1 is a model_org or a model_name directory
        is_model_org = False
        # Heuristic: if a subdirectory contains other directories, we assume it's an org folder.
        # A better check might be needed if folder names can be ambiguous.
        sub_entries = os.listdir(path1)
        if sub_entries:
            first_sub_path = os.path.join(path1, sub_entries[0])
            if os.path.isdir(first_sub_path):
                # Check if the sub-sub-directory contains parquet files.
                # If not, we assume path1 is a model_org.
                try:
                    sub_sub_entries = os.listdir(first_sub_path)
                    if not any(f.endswith('.parquet') for f in sub_sub_entries):
                        is_model_org = True
                except NotADirectoryError:
                    pass # sub_sub_entries is a file

        if is_model_org:
            model_org = entry
            for model_name_str in os.listdir(path1):
                path2 = os.path.join(path1, model_name_str)
                if not os.path.isdir(path2):
                    continue
                full_model_name = f"{model_org}_{model_name_str}"
                for eval_time in os.listdir(path2):
                    path3 = os.path.join(path2, eval_time)
                    if os.path.isdir(path3):
                        process_evaluation_directory(path3, full_model_name, eval_time, files_by_eval)
        else:
            model_name = entry
            for eval_time in os.listdir(path1):
                path2 = os.path.join(path1, eval_time)
                if os.path.isdir(path2):
                    process_evaluation_directory(path2, model_name, eval_time, files_by_eval)

    if not files_by_eval:
        print("No parquet files found to process.")
        return

    print("\nFound the following evaluation types:")
    for eval_name in sorted(files_by_eval.keys()):
        print(f"- {eval_name}")

    # 2. Second pass: Copy files to organized directory structure
    print("\n--- Organizing files by evaluation type ---")
    import shutil
    
    for eval_name, file_infos in tqdm(files_by_eval.items(), desc="Organizing files"):
        if not file_infos:
            continue
            
        # Create evaluation type directory
        safe_eval_name = eval_name.replace('|', '_').replace(':', '_')
        eval_dir = os.path.join(output_dir, safe_eval_name)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Process and save files for each model
        for info in file_infos:
            try:
                # Load parquet file and add metadata columns
                df = pd.read_parquet(info['path'])
                df['model_name'] = info['model_name']
                df['eval_name'] = eval_name
                # Extract timestamp from evaluation_time (assuming format like 2025-07-01T09-34-12.763860)
                timestamp = info['evaluation_time']
                if '_' in timestamp:  # Remove any prefix if present
                    timestamp = timestamp.split('_')[-1]
                df['timestamp'] = timestamp
                
                output_filename = f"{info['model_name']}.parquet"
                output_path = os.path.join(eval_dir, output_filename)
                df.to_parquet(output_path, index=False)
                
                # Free memory
                del df
            except Exception as e:
                print(f"Error processing file {info['path']}: {e}")
                
    print("\nFiles have been organized in the following structure:")
    print(f"{output_dir}/")
    print("└── [evaluation_type]/")
    print("    └── [model_name].parquet")

def process_evaluation_directory(eval_dir_path, model_name, eval_time, files_by_eval):
    for filename in os.listdir(eval_dir_path):
        if filename.endswith(".parquet"):
            file_path = os.path.join(eval_dir_path, filename)
            try:
                # Extract evaluation name from filename
                # Example: details_community|cybersec_eval:cybersecurity_roadmap|0_2025-07-01T09-36-01.393495.parquet
                parts = filename.split('|')
                if len(parts) >= 2:
                    eval_name = parts[1].split('|')[0]  # Get the middle part (e.g., cybersec_eval:cybersecurity_roadmap)
                else:
                    # Fallback to old method if format is different
                    eval_name_part = filename.split('_')[0]
                    if eval_name_part.startswith("details_"):
                        eval_name = eval_name_part[len("details_"):]
                    else:
                        eval_name = eval_name_part
                
                # Store file metadata, not the data itself
                files_by_eval[eval_name].append({
                    "path": file_path,
                    "model_name": model_name,
                    "evaluation_time": eval_time
                })
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine evaluation results from a nested directory structure in a memory-efficient way.")
    parser.add_argument("folder_path", type=str, help="The root folder containing the evaluation results.")
    parser.add_argument("-o", "--output_dir", type=str, default="results/combined_details", help="The directory to save the combined parquet files.")
    
    args = parser.parse_args()
    
    combine_results_memory_efficient(args.folder_path, args.output_dir)
