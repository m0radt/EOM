# split_data.py
import json
import random
from pathlib import Path

from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
)
def split_and_save(
    dataset: str, out_prefix: str,
    split_ratio: float = 0.8, val_size: int = None, seed: int = 42,
    mini: bool = False, noextreme: bool = False, version: str = "default",
):
    if dataset not in {"humaneval", "mbpp"}:
        raise ValueError("dataset must be 'humaneval' or 'mbpp'")

    if dataset == "humaneval":
        problems = get_human_eval_plus(mini=mini, noextreme=noextreme, version=version)
    else:
        problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)

    task_ids = sorted(problems.keys())
    rng = random.Random(seed)
    rng.shuffle(task_ids)

    if val_size is not None:
        if val_size <= 0 or val_size >= len(task_ids):
            raise ValueError(f"val_size must be between 1 and {len(task_ids)-1}")
        cut = val_size
    else:
        cut = int(len(task_ids) * split_ratio)

    val_ids = task_ids[:cut]
    test_ids = task_ids[cut:]

    assert len(val_ids) + len(test_ids) == len(task_ids), "Split sizes do not add up"

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    val_file = out_prefix.with_name(out_prefix.stem + "_val_ids.json")
    test_file = out_prefix.with_name(out_prefix.stem + "_test_ids.json")

    if val_file.exists() and test_file.exists():
        print(f"Split already exists: {val_file}, {test_file} → skipping.")
        return
    
    with open(val_file, "w") as f:
        json.dump(sorted(val_ids, key=lambda x: int(x.split("/")[1])), f, indent=2)
    with open(test_file, "w") as f:
        json.dump(sorted(test_ids, key=lambda x: int(x.split("/")[1])), f, indent=2)

    print(f"\nSaved {len(val_ids)} val IDs to {val_file}")
    print(f"Saved {len(test_ids)} test IDs to {test_file}\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["humaneval","mbpp"])
    p.add_argument("--out_prefix", required=True, help="e.g., splits/humaneval_split")
    p.add_argument("--split_ratio", type=float, default=0.8)
    p.add_argument("--val_size", type=int, default=None,
                   help="Number of tasks for validation (overrides split_ratio if set)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini", action="store_true")
    p.add_argument("--noextreme", action="store_true")
    p.add_argument("--version", default="default")
    args = p.parse_args()

    split_and_save(
        dataset=args.dataset,
        out_prefix=args.out_prefix,
        split_ratio=args.split_ratio,
        val_size=args.val_size,
        seed=args.seed,
        mini=args.mini,
        noextreme=args.noextreme,
        version=args.version,
    )
