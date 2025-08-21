#!/usr/bin/env python
"""
Script to download the fall detection dataset from Kaggle
"""
import os
import sys
import kagglehub
import argparse
from pathlib import Path

def download_dataset(output_dir=None):
    """
    Download the fall detection dataset from Kaggle
    
    Args:
        output_dir: Directory to save the dataset to. If None, will use 'datasets/fall_detection'
    
    Returns:
        Path to the downloaded dataset
    """
    if output_dir is None:
        # Create a datasets directory in the project root
        project_root = Path(__file__).parent.absolute()
        output_dir = project_root / "datasets" / "fall_detection"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading fall detection dataset to {output_dir}...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download(
            "uttejkumarkandagatla/fall-detection-dataset"
        )
        
        # Move or copy files to the desired output directory
        import shutil
        if os.path.isdir(path):
            # If path is a directory, copy its contents
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        else:
            # If path is a file, copy it directly
            shutil.copy2(path, output_dir)
            
        print(f"Dataset downloaded successfully to: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nNote: You may need to authenticate with Kaggle first.")
        print("To set up Kaggle authentication:")
        print("1. Install the Kaggle CLI: pip install kaggle")
        print("2. Create a Kaggle account at https://www.kaggle.com if you don't have one")
        print("3. Go to your Kaggle account settings > API section and click 'Create New API Token'")
        print("4. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("6. Try running this script again")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download fall detection dataset from Kaggle")
    parser.add_argument("--output", "-o", type=str, help="Directory to save the dataset to")
    args = parser.parse_args()
    
    path = download_dataset(args.output)
    
    if path:
        print("\nDataset is ready for use!")
        print("You can now use the dataset in your fall detection application.")

if __name__ == "__main__":
    main()
