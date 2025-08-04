#!/usr/bin/env python3
"""
Script to convert JSONL files to Parquet format.
This is useful when parquet files are ignored by git and you need to regenerate them on a new machine.

Usage:
    python convert_jsonl_to_parquet.py [input_file.jsonl] [output_file.parquet]
    python convert_jsonl_to_parquet.py  # converts train.jsonl and test.jsonl by default
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm


def read_jsonl(file_path: str) -> list[dict]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    print(f"Reading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading lines"), 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                    continue
    
    print(f"Successfully loaded {len(data)} records from {file_path}")
    return data


def convert_jsonl_to_parquet(jsonl_path: str, parquet_path: str = None) -> None:
    """Convert a JSONL file to Parquet format."""
    
    # Generate output path if not provided
    if parquet_path is None:
        jsonl_file = Path(jsonl_path)
        parquet_path = jsonl_file.with_suffix('.parquet')
    
    # Check if input file exists
    if not Path(jsonl_path).exists():
        print(f"Error: Input file {jsonl_path} does not exist!")
        return
    
    # Read JSONL data
    data = read_jsonl(jsonl_path)
    
    if not data:
        print(f"Warning: No data found in {jsonl_path}")
        return
    
    # Convert to DataFrame
    print(f"Converting to DataFrame...")
    df = pd.DataFrame(data)
    
    # Display basic info about the data
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample data
    print("\nSample data:")
    for col in df.columns[:3]:  # Show first 3 columns
        if col in df.columns:
            sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
            # Truncate long strings for display
            if isinstance(sample_val, str) and len(sample_val) > 50:
                sample_val = sample_val[:50] + "..."
            print(f"  {col}: {sample_val}")
    
    # Save to Parquet
    print(f"Saving to Parquet: {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    
    # Verify file was created and show size comparison
    parquet_size = Path(parquet_path).stat().st_size
    jsonl_size = Path(jsonl_path).stat().st_size
    compression_ratio = (1 - parquet_size / jsonl_size) * 100
    
    print(f"✅ Conversion completed!")
    print(f"   Original JSONL: {jsonl_size / 1024 / 1024:.2f} MB")
    print(f"   Parquet file:   {parquet_size / 1024 / 1024:.2f} MB")
    print(f"   Compression:    {compression_ratio:.1f}% smaller")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL files to Parquet format")
    parser.add_argument("input", nargs="?", help="Input JSONL file path")
    parser.add_argument("output", nargs="?", help="Output Parquet file path (optional)")
    parser.add_argument("--batch", action="store_true", help="Convert train.jsonl and test.jsonl")
    
    args = parser.parse_args()
    
    if args.batch or (not args.input):
        # Default batch mode: convert train.jsonl and test.jsonl
        files_to_convert = ["train.jsonl", "test.jsonl"]
        
        for file_path in files_to_convert:
            if Path(file_path).exists():
                print(f"\n{'='*60}")
                print(f"Converting {file_path}")
                print('='*60)
                convert_jsonl_to_parquet(file_path)
            else:
                print(f"Skipping {file_path} (file not found)")
    else:
        # Single file mode
        convert_jsonl_to_parquet(args.input, args.output)


if __name__ == "__main__":
    main() 