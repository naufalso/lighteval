import os
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm


def combine_json_results_summary(input_path, output_file=None):
    """
    Combines JSON evaluation results into a summary CSV with main metrics only.
    
    Expected structure:
    input_path/
    ├── models_[model_name]/
    │   └── results_[timestamp].json
    └── [org_name]/
        └── [model_name]/
            └── results_[timestamp].json
    
    Args:
        input_path (str): Path to directory containing model result folders
        output_file (str, optional): Path for output file. If None, saves to input_path/combined_results_summary.csv
    """
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # Collect all JSON files
    json_files = []
    
    print("Scanning for JSON result files...")
    for entry in tqdm(list(input_path.iterdir())):
        if not entry.is_dir():
            continue
            
        # Look for JSON files directly in this directory (flat structure)
        json_files_in_dir = list(entry.glob('results_*.json'))
        
        if json_files_in_dir:
            # This is a model directory with JSON files
            model_name = entry.name.replace('models_', '') if entry.name.startswith('models_') else entry.name
            for json_file in json_files_in_dir:
                json_files.append({
                    'path': json_file,
                    'model_name': model_name,
                    'filename': json_file.name
                })
        else:
            # Check if this is an organization directory with nested model directories
            for sub_entry in entry.iterdir():
                if sub_entry.is_dir():
                    sub_json_files = list(sub_entry.glob('results_*.json'))
                    if sub_json_files:
                        model_name = f"{entry.name}/{sub_entry.name}"
                        for json_file in sub_json_files:
                            json_files.append({
                                'path': json_file,
                                'model_name': model_name,
                                'filename': json_file.name
                            })
    
    if not json_files:
        print("No JSON result files found in the specified directory.")
        return
    
    print(f"Found {len(json_files)} JSON result files")
    
    # Process JSON files and combine results
    combined_data = []
    
    print("Processing JSON files...")
    for file_info in tqdm(json_files):
        try:
            with open(file_info['path'], 'r') as f:
                data = json.load(f)
            
            # Extract timestamp from filename
            timestamp = file_info['filename'].replace('results_', '').replace('.json', '')
            
            # Extract general config info
            config_general = data.get('config_general', {})
            model_name = file_info['model_name']
            
            # Process results - only extract acc_norm metrics
            results = data.get('results', {})
            for task_name, metrics in results.items():
                acc_norm = metrics.get('acc_norm')
                acc_norm_stderr = metrics.get('acc_norm_stderr')
                
                if acc_norm is not None:
                    row = {
                        'model_name': model_name,
                        'timestamp': timestamp,
                        'task_name': task_name,
                        'acc_norm': acc_norm,
                        'acc_norm_stderr': acc_norm_stderr,
                        'evaluation_time_seconds': config_general.get('total_evaluation_time_secondes'),
                        'lighteval_sha': config_general.get('lighteval_sha'),
                        'num_fewshot_seeds': config_general.get('num_fewshot_seeds'),
                        'max_samples': config_general.get('max_samples')
                    }
                    combined_data.append(row)
                        
        except Exception as e:
            print(f"Error processing file {file_info['path']}: {e}")
            continue
    
    if not combined_data:
        print("No valid data found in JSON files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Sort by model name, task name, and timestamp
    df = df.sort_values(['model_name', 'task_name', 'timestamp'])
    
    # Set output file path
    if output_file is None:
        output_file = input_path / 'combined_results_summary.csv'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Combined results saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total rows: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Unique tasks: {df['task_name'].nunique()}")
    print(f"Unique timestamps: {df['timestamp'].nunique()}")
    
    # Show sample of the data
    print(f"\nSample data:")
    print(df.head(10))
    
    return df


def create_pivot_summary(df=None, input_path=None, output_file=None):
    """
    Creates a pivot table summary with models as rows and tasks as columns.
    
    Args:
        df (pd.DataFrame, optional): DataFrame from combine_json_results_summary
        input_path (str, optional): Path to load combined_results_summary.csv if df not provided
        output_file (str, optional): Path for output file
    """
    
    if df is None:
        if input_path is None:
            raise ValueError("Either df or input_path must be provided")
        
        input_path = Path(input_path)
        csv_file = input_path / 'combined_results_summary.csv'
        
        if not csv_file.exists():
            raise ValueError(f"Combined results file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
    
    # Create pivot table with latest results for each model-task combination
    df_latest = df.sort_values('timestamp').groupby(['model_name', 'task_name']).tail(1)
    
    pivot_df = df_latest.pivot(index='model_name', columns='task_name', values='acc_norm')
    
    # Set output file path
    if output_file is None:
        if input_path:
            output_file = Path(input_path) / 'results_pivot_summary.csv'
        else:
            output_file = 'results_pivot_summary.csv'
    
    # Save pivot table
    pivot_df.to_csv(output_file)
    print(f"Pivot summary saved to: {output_file}")
    
    # Print summary
    print(f"\nPivot Summary:")
    print(f"Models: {len(pivot_df)}")
    print(f"Tasks: {len(pivot_df.columns)}")
    print("\nSample pivot data:")
    print(pivot_df.head())
    
    return pivot_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine JSON evaluation results into a summary CSV with main metrics only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - combine results from a directory
  python combine_results_summary.py /path/to/results

  # Specify custom output file
  python combine_results_summary.py /path/to/results -o my_results.csv

  # Create both summary and pivot table
  python combine_results_summary.py /path/to/results --pivot

  # Only create pivot table from existing summary
  python combine_results_summary.py /path/to/results --pivot-only

Expected directory structure:
  input_path/
  ├── models_[model_name]/
  │   └── results_[timestamp].json
  └── [org_name]/
      └── [model_name]/
          └── results_[timestamp].json
        """
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to directory containing model result folders"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path for combined results (default: input_path/combined_results_summary.csv)"
    )
    
    parser.add_argument(
        "--pivot",
        action="store_true",
        help="Also create a pivot table summary with models as rows and tasks as columns"
    )
    
    parser.add_argument(
        "--pivot-only",
        action="store_true",
        help="Only create pivot table from existing combined_results_summary.csv (skip main processing)"
    )
    
    parser.add_argument(
        "--pivot-output",
        type=str,
        default=None,
        help="Output file path for pivot table (default: input_path/results_pivot_summary.csv)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pivot_only and args.output:
        print("Warning: --output is ignored when using --pivot-only")
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        exit(1)
    
    # Main processing
    df = None
    
    if not args.pivot_only:
        # Combine JSON results into summary CSV
        print(f"Processing JSON files from: {input_path}")
        df = combine_json_results_summary(input_path, args.output)
        
        if df is None:
            print("No data was processed. Exiting.")
            exit(1)
    
    # Create pivot summary if requested
    if args.pivot or args.pivot_only:
        if args.pivot_only:
            # Load existing CSV file
            csv_file = input_path / 'combined_results_summary.csv'
            if not csv_file.exists():
                print(f"Error: No existing combined results file found at {csv_file}")
                print("Run without --pivot-only first to generate the summary file.")
                exit(1)
            print(f"Loading existing summary from: {csv_file}")
        
        pivot_df = create_pivot_summary(
            df=df,
            input_path=input_path,
            output_file=args.pivot_output
        )
        
        if pivot_df is None:
            print("Failed to create pivot summary.")
            exit(1)