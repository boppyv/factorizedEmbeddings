# test_modal.py
import modal

app = modal.App("test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)

@app.function(gpu="any", image=image, timeout=60)
def gpu_check():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return "success"

@app.local_entrypoint()
def main():
    result = gpu_check.remote()
    print(result)
