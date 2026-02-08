# test_all_local.py
import subprocess
import sys

COMMON = [
    "--dataset=shakespeare",
    "--n_layer=2", "--n_head=2", "--n_embd=64",
    "--block_size=64", "--batch_size=4",
    "--max_iters=50", "--eval_interval=25", "--eval_iters=5",
    "--device=cpu", "--compile=False", "--wandb_log=False",
]

CONFIGS = [
    {"embedding_type": "standard"},
    {"embedding_type": "factorized", "embedding_K": 16},
    {"embedding_type": "generator", "embedding_K": 16},
    {"embedding_type": "progressive", "embedding_K": 16},
    {"embedding_type": "growing", "embedding_K": 16},
]

passed = []
failed = []

for cfg in CONFIGS:
    name = cfg["embedding_type"]
    args = ["python", "train.py"] + COMMON
    for k, v in cfg.items():
        args.append(f"--{k}={v}")

    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    result = subprocess.run(args)

    if result.returncode != 0:
        print(f"FAILED: {name}")
        failed.append(name)
    else:
        print(f"PASSED: {name}")
        passed.append(name)

print(f"\n{'='*50}")
print(f"Results: {len(passed)} passed, {len(failed)} failed")
if passed:
    print(f"  Passed: {', '.join(passed)}")
if failed:
    print(f"  Failed: {', '.join(failed)}")
    sys.exit(1)
print("All good!")
