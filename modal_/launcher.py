# modal_/launcher.py
import modal
import os

volume = modal.Volume.from_name("nanoGPT-data", create_if_missing=True)
results_volume = modal.Volume.from_name("nanoGPT-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "wandb",
    )
    .apt_install("git")
)

app = modal.App("embedding-experiments")

@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume, "/results": results_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=14400,
)
def train_run(
    embedding_type: str = "standard",
    embedding_K: int = 64,
    seed: int = 42,
    wandb_run_name: str = "run",
    max_iters: int = 600000,
    eval_interval: int = 2000,
):
    import subprocess

    subprocess.run([
        "git", "clone", "https://github.com/boppyv/factorizedEmbeddings.git",
        "/root/project"
    ], check=True)
    os.chdir("/root/project")

    os.makedirs("data/openwebtext", exist_ok=True)
    if not os.path.exists("data/openwebtext/train.bin"):
        os.symlink("/data/train.bin", "data/openwebtext/train.bin")
        os.symlink("/data/val.bin", "data/openwebtext/val.bin")

    subprocess.run([
        "python", "train.py",
        f"--embedding_type={embedding_type}",
        f"--embedding_K={embedding_K}",
        f"--dataset=openwebtext",
        f"--wandb_log=True",
        f"--wandb_project=embedding-sharing",
        f"--wandb_run_name={wandb_run_name}",
        "--compile=True",
        
        # single-GPU training settings
        "--gradient_accumulation_steps=5",
        "--max_iters=50000",
        "--lr_decay_iters=50000",
        "--warmup_iters=500",
        "--eval_interval=1000",
    ], check=True)

@app.local_entrypoint()
def main(
    embedding_type: str = "standard",
    embedding_k: int = 64,
    seeds: str = "42",
    parallel: bool = False,
):
    seed_list = [int(s) for s in seeds.split(",")]

    for seed in seed_list:
        run_name = f"{embedding_type}_K{embedding_k}_seed{seed}"
        if parallel:
            train_run.spawn(embedding_type, embedding_k, seed, run_name)
        else:
            train_run.remote(embedding_type, embedding_k, seed, run_name)

    print("All runs launched!")
