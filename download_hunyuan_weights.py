import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

print("Checking and downloading tencent/Hunyuan3D-2mv...")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Download model.fp16.safetensors and config.yaml
try:
    print("Downloading config.yaml...")
    cfg = hf_hub_download(
        repo_id="tencent/Hunyuan3D-2mv",
        filename="hunyuan3d-dit-v2-mv/config.yaml",
        resume_download=True
    )
    print(f"config.yaml ready: {cfg}")

    print("Downloading model.fp16.safetensors (4.7 GB)...")
    model_path = hf_hub_download(
        repo_id="tencent/Hunyuan3D-2mv",
        filename="hunyuan3d-dit-v2-mv/model.fp16.safetensors",
        resume_download=True
    )
    print(f"model.fp16.safetensors ready: {model_path}")

    # Also download VAE if needed by hy3dgen
    print("Downloading hunyuan3d-vae-v2-0 config and model...")
    vae_cfg = hf_hub_download(
        repo_id="tencent/Hunyuan3D-2",
        filename="hunyuan3d-vae-v2-0/config.yaml",
        resume_download=True
    )
    vae_model = hf_hub_download(
        repo_id="tencent/Hunyuan3D-2",
        filename="hunyuan3d-vae-v2-0/model.safetensors",
        resume_download=True
    )
    print("All required Hunyuan3D-2/2mv weights downloaded successfully!")
except Exception as e:
    print(f"Error downloading: {e}")
    sys.exit(1)
