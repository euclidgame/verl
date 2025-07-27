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
Preprocess the `xiaomama2002/olympic_dataset` (Olympiad-style math problems)
into the parquet format expected by VeRL.

The resulting directory structure is identical to the other examples:

    <local_dir>/train.parquet
    <local_dir>/test.parquet   # or validation.parquet if the source dataset
                               # provides a ``validation`` split instead of
                               # ``test``.

Each row contains at least the following fields expected by VeRL
training scripts:

    data_source : str  – the original HF repo id (used by reward fn)
    prompt      : list – chat-like prompt messages
    ability     : str  – coarse tag ("math")
    reward_model: dict – information for rule-based reward computation
    extra_info  : dict – arbitrary metadata helpful for debugging

Run with
```
python examples/data_preprocess/olympic_dataset.py \
       --local_dir ~/data/olympic_dataset
```
Optionally add ``--hdfs_dir hdfs://…`` to copy the parquet files to HDFS.
"""

import argparse
import os
from typing import Dict, Any

import datasets

from verl.utils.hdfs_io import copy, makedirs

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_map_fn(split: str, data_source: str):
    """Return a datasets.map-compatible processing function."""

    def _process_fn(example: Dict[str, Any], idx: int):
        # The source dataset is assumed to expose at least "question" and
        # "answer" fields.  If field names differ an explicit mapping is
        # required.
        question_raw = example.pop("question")  # raises KeyError if missing

        # Instruction that encourages chain-of-thought style reasoning.
        prompt_text  = f"""
Your task is to write a proof solution to the following problem. Your proof will be graded by judges for correctness and completeness. When you write your proof, follow these guidelines:
  - You are creating a proof, not a proof outline. Each step should be carefully explained and documented. If not properly explained, the judge will assume that you cannot explain it, and therefore decrease your grade.
  - You can use general theorems and lemmas, but only if they are well-known. As a rule of thumb: if the result has a name and is famous enough to have a Wikipedia page or something similar to describe it, it is allowed. Any result from papers that would not be taught in high school or low-level bachelor courses in mathematics should not be used. Any use of such results will immediately give you a zero grade.
  - Do not skip computation steps in your proof. Clearly explain what transformations were done and why they are allowed in each step of a calculation.
  - You should use correct LaTeX notation to write equations and mathematical symbols. You should encompass these equations in appropriate symbols ("\\(" and "\\)" for inline math, "\\[" and "\\]" for block math) to enhance the clarity of your proof. Do not use any unicode characters.
  - Your proof should be self-contained.
  - If you are not sure about a specific step, or do not know how to prove an intermediate result, clearly state this. It is much preferable to indicate your uncertainty rather than making incorrect statements or claims.

PROBLEM: {question_raw}
"""

        # Build a single-turn prompt (user only).  Add a *system* role if you
        # prefer the multi-turn style.
        # reward will be computed by external service; ground-truth extraction removed

        processed = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "llm_as_a_judge", "ground_truth": "0"},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question_raw,
            },
        }
        return processed

    return _process_fn


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess olympic dataset for VeRL.")
    parser.add_argument("--local_dir", default="~/data/olympic_dataset", help="Where to save the parquet files.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS path to copy the parquet files to.")

    args = parser.parse_args()
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir  = args.hdfs_dir

    # ------------------------------------------------------------------
    # Load source dataset from Hugging Face Hub
    # ------------------------------------------------------------------
    data_source = "xiaomama2002/olympic_dataset"
    print(f"Loading dataset {data_source} …")
    dataset = datasets.load_dataset(data_source)

    # Determine split names present in the dataset
    # DatasetDict is iterable; convert to list of split names
    splits = list(dataset.keys())  # type: ignore[arg-type]
    print(f"Found splits: {splits}")

    # ------------------------------------------------------------------
    # Map each split to VeRL format and write parquet
    # ------------------------------------------------------------------
    os.makedirs(local_dir, exist_ok=True)

    for split_name in splits:
        print(f"Processing split: {split_name}")
        hf_split = dataset[split_name]  # type: ignore[index]
        mapped   = hf_split.map(  # type: ignore[attr-defined]
            function=_make_map_fn(str(split_name), data_source),
            with_indices=True,
        )

        out_path = os.path.join(local_dir, f"{split_name}.parquet")
        print(f"Writing {out_path}")
        mapped.to_parquet(out_path)

    # If the original dataset uses "validation" split, you may want to keep a
    # copy named "test.parquet" for consistency with other examples.
    if "validation" in splits and "test" not in splits:
        src = os.path.join(local_dir, "validation.parquet")
        dst = os.path.join(local_dir, "test.parquet")
        print(f"Duplicating validation → test: {src} -> {dst}")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)

    # ------------------------------------------------------------------
    # Optional HDFS copy
    # ------------------------------------------------------------------
    if hdfs_dir is not None:
        print(f"Copying parquet files to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 