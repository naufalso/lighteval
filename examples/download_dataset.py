from datasets import load_dataset
from datasets import get_dataset_config_names

import argparse

def download_dataset(dataset_name: str, dataset_subset: str, split: str = 'test'):
    """
    Downloads a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset to download.
        dataset_subset (str): The specific subset of the dataset to download.
        split (str): The split of the dataset to download (default is 'test').

    Returns:
        Dataset: The downloaded dataset.
    """
    return load_dataset(dataset_name, dataset_subset, split=split)

class DatasetDownloader:
    def __init__(self, dataset_name, dataset_subsets, split='test'):
        self.dataset_name = dataset_name
        self.dataset_subsets = dataset_subsets
        self.split = split

    def download(self):
        datasets = {}
        if self.dataset_subsets is None or len(self.dataset_subsets) == 0:
            raise ValueError("No dataset subsets provided. Please specify at least one subset to download.")

        if self.dataset_subsets[0] == 'all':
            self.dataset_subsets = get_dataset_config_names(self.dataset_name)

        for subset in self.dataset_subsets:
            print(f"Downloading {self.dataset_name} - {subset} ({self.split})...")
            datasets[subset] = download_dataset(self.dataset_name, subset, self.split)
            print(f"Downloaded {subset} with {len(datasets[subset])} samples.")
        return datasets

def main():
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub.")
    parser.add_argument('dataset_name', type=str, help='Name of the dataset to download')
    parser.add_argument('dataset_subsets', type=str, nargs='+', help='List of dataset subsets to download')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to download (default: test)')
    args = parser.parse_args()

    downloader = DatasetDownloader(args.dataset_name, args.dataset_subsets, args.split)
    downloader.download()

if __name__ == "__main__":
    main()