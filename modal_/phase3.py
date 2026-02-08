# modal_/phase3.py
from modal_.launcher import train_run, app

@app.local_entrypoint()
def main_phase3():
    embedding_types = []
    embedding_Ks = []
    seeds = []
    names = []

    for emb_type in ["standard", "factorized", "generator", "progressive"]:
        for seed in [42, 123, 456]:
            embedding_types.append(emb_type)
            embedding_Ks.append(64)
            seeds.append(seed)
            names.append(f"{emb_type}_K64_seed{seed}_phase3")

    print(f"Launching {len(names)} runs...")
    train_run.spawn_map(embedding_types, embedding_Ks, seeds, names)
    print("All runs submitted. Safe to close terminal.")
