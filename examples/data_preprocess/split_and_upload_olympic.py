# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Split the single-train dataset `xiaomama2002/olympic_dataset` into
train/test and upload to a new Hugging Face repo.

Run with
```
python examples/data_preprocess/split_and_upload_olympic.py \
       --new_repo_id <your_username>/olympic_split \
       --test_size 0.2 \
       --hf_token hf_xxxxxxxx
```

The script will shuffle the original train split, take ``test_size`` fraction
for the test set, and push the result to the new repo.  If the repo doesn’t
exist it will be created (private by default).
"""

import argparse
import os
import random
from typing import Dict

import datasets
from datasets import DatasetDict

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and upload olympic dataset.")
    parser.add_argument("--original_repo", default="xiaomama2002/olympic_dataset",
                        help="Hugging Face repo ID to download. (default: %(default)s)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of train to move to test split. (default: %(default)s)")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                        help="Hugging Face API token (or set HF_TOKEN env var).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling. (default: %(default)s)")
    parser.add_argument("--private", action="store_true",
                        help="Create the new repo as private (default: public).")

    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("HF_TOKEN environment variable or --hf_token is required for upload")

    print(f"Loading original dataset: {args.original_repo}")
    original = datasets.load_dataset(args.original_repo)

    if "train" not in original:
        raise ValueError(f"Original dataset has no 'train' split: {list(original.keys())}")

    train_split = original["train"]

    print(f"Shuffling with seed {args.seed}")
    random.seed(args.seed)
    shuffled = train_split.shuffle(seed=args.seed)

    print(f"Splitting into train/test with test_size={args.test_size}")
    split_dict: Dict[str, datasets.Dataset] = shuffled.train_test_split(test_size=args.test_size)

    new_dataset = DatasetDict({
        "train": split_dict["train"],
        "test": split_dict["test"],
    })

    print(f"Uploading to {args.original_repo} (private={args.private})")
    new_dataset.push_to_hub(args.original_repo, private=args.private, token=args.hf_token)

    print(f"Upload complete. New dataset available at https://huggingface.co/{args.original_repo}") 