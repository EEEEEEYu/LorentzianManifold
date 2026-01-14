"""
Download script for Kinetics400 dataset
Standalone script for downloading and unzipping Kinetics400
"""

import os
import argparse
from pathlib import Path
import subprocess


def download_kinetics400(root_dir, split='train'):
    """
    Download Kinetics400 dataset
    
    Note: Kinetics400 requires downloading videos from YouTube.
    This is a placeholder script that shows the structure.
    For actual download, you may need to use:
    - official Kinetics downloader scripts
    - or use pre-processed datasets
    
    Args:
        root_dir: Root directory for dataset
        split: 'train' or 'val'
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Kinetics400 dataset download script")
    print(f"Target directory: {root_dir}")
    print(f"Split: {split}")
    print()
    print("Note: Kinetics400 download requires:")
    print("1. Official downloader scripts from DeepMind")
    print("2. YouTube video access")
    print("3. Significant storage space (~450GB for full dataset)")
    print()
    print("For actual implementation, refer to:")
    print("https://github.com/deepmind/kinetics-i3d")
    print("or use pre-processed versions from:")
    print("- Something-Something V2 (220K videos)")
    print("- Other video action recognition datasets")
    print()
    print("This script creates the directory structure.")
    print("Replace this with actual download logic when ready.")
    
    # Create directory structure
    split_dir = root_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {split_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Kinetics400 dataset')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory for dataset')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split to download')
    
    args = parser.parse_args()
    download_kinetics400(args.root_dir, args.split)
