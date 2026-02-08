# scripts/upload_data.py
import modal

volume = modal.Volume.from_name("nanoGPT-data", create_if_missing=True)
app = modal.App("upload-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "tiktoken", "datasets")
)

@app.function(volumes={"/data": volume}, image=image, timeout=3600)
def upload():
    import subprocess
    import os

    # Clone your repo and prepare data on the remote machine
    # (faster than uploading 17GB from your local machine)
    subprocess.run(["git", "clone", "https://github.com/boppyv/factorizedEmbeddings.git"], check=True)
    os.chdir("factorizedEmbeddings/data/openwebtext")
    subprocess.run(["python", "prepare.py"], check=True)

    # Copy prepared data to the persistent volume
    subprocess.run(["cp", "train.bin", "/data/train.bin"], check=True)
    subprocess.run(["cp", "val.bin", "/data/val.bin"], check=True)
    print("Data uploaded to volume")

@app.local_entrypoint()
def main():
    upload.remote()
