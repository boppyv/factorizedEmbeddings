# modal_/phase3.py
from modal_.launcher import train_run, app

SEEDS = [42, 123, 456]

EXPERIMENTS = [
    {"embedding_type": "standard", "embedding_K": 64},
    {"embedding_type": "factorized", "embedding_K": 64},
    {"embedding_type": "progressive", "embedding_K": 64},
]

@app.local_entrypoint()
def main():
    futures = []
    for cfg in EXPERIMENTS:
        for seed in SEEDS:
            name = f"{cfg['embedding_type']}_K{cfg['embedding_K']}_seed{seed}"
            futures.append(train_run.spawn(
                embedding_type=cfg["embedding_type"],
                embedding_K=cfg["embedding_K"],
                seed=seed,
                wandb_run_name=name,
            ))
            print(f"Launched: {name}")

    print(f"\n{len(futures)} runs launched on separate A100s.")
    print("All runs use .spawn() — safe to close your terminal.")
    print("Monitor progress at https://wandb.ai and https://modal.com")
