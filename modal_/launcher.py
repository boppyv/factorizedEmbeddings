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
    gpu="A100",
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
):
    import subprocess

    # Clone your repo fresh each time (or cache it in a volume later)
    subprocess.run([
        "git", "clone", "https://github.com/boppyv/factorizedEmbeddings.git",
        "/root/project"
    ], check=True)
    os.chdir("/root/project")

    # Symlink data from the persistent volume
    os.makedirs("data/shakespeare", exist_ok=True)
    # (for now — later swap to openwebtext)

    # Run training with command-line overrides, same as local
    subprocess.run([
        "python", "train.py",
        f"--embedding_type={embedding_type}",
        f"--embedding_K={embedding_K}",
        f"--wandb_log=True",
        f"--wandb_project=embedding-sharing",
        f"--wandb_run_name={wandb_run_name}",
        "--compile=True",
    ], check=True)

    # Copy checkpoint to persistent storage
    run_name = f"{embedding_type}_K{embedding_K}_seed{seed}"
    subprocess.run(["cp", "-r", "out/", f"/results/{run_name}/"], check=True)
    results_volume.commit()
    print(f"Run {run_name} complete")


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
