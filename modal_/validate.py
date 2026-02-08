# modal_/validate.py
from modal_.launcher import train_run, app

CONFIGS = [
    {"embedding_type": "standard", "embedding_K": 64},
    {"embedding_type": "factorized", "embedding_K": 64},
    {"embedding_type": "generator", "embedding_K": 64},
    {"embedding_type": "progressive", "embedding_K": 64},
]

@app.local_entrypoint()
def main_validate():
    futures = []
    for cfg in CONFIGS:
        name = f"validate_{cfg['embedding_type']}_K{cfg['embedding_K']}"
        futures.append(train_run.spawn(
            embedding_type=cfg["embedding_type"],
            embedding_K=cfg["embedding_K"],
            seed=42,
            wandb_run_name=name,
            max_iters=500,
            eval_interval=100,
        ))
        print(f"Launched: {name}")

    print(f"\n{len(futures)} runs launched. Waiting...")
    for f in futures:
        f.get()
    print("All done!")
