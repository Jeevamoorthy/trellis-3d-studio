import os
import sys
import time
import requests
from pathlib import Path

# URL for model.fp16.safetensors
URL = "https://huggingface.co/tencent/Hunyuan3D-2mv/resolve/main/hunyuan3d-dit-v2-mv/model.fp16.safetensors"
CONFIG_URL = "https://huggingface.co/tencent/Hunyuan3D-2mv/resolve/main/hunyuan3d-dit-v2-mv/config.yaml"

# Target local path where hy3dgen checks
LOCAL_DIR = Path(r"C:\Users\JEEVA\.cache\hy3dgen\tencent\Hunyuan3D-2mv\hunyuan3d-dit-v2-mv")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = LOCAL_DIR / "model.fp16.safetensors"
CONFIG_FILE = LOCAL_DIR / "config.yaml"

# First download config.yaml
if not CONFIG_FILE.exists() or CONFIG_FILE.stat().st_size == 0:
    print(f"Downloading config.yaml to {CONFIG_FILE}...")
    r = requests.get(CONFIG_URL)
    with open(CONFIG_FILE, "wb") as f:
        f.write(r.content)
    print("config.yaml saved.")
else:
    print("config.yaml already exists.")

# Check if there is an existing incomplete file from HF cache to reuse!
hf_blobs = list(Path(r"C:\Users\JEEVA\.cache\huggingface\hub\models--tencent--Hunyuan3D-2mv\blobs").glob("*.incomplete"))
if not MODEL_FILE.exists() and hf_blobs:
    largest_blob = max(hf_blobs, key=lambda f: f.stat().st_size)
    if largest_blob.stat().st_size > 50 * 1024 * 1024:
        print(f"Adopting previously downloaded {largest_blob.stat().st_size / (1024*1024):.1f} MB from HF cache...")
        import shutil
        shutil.copyfile(str(largest_blob), str(MODEL_FILE))
        print("Adoption complete. Will resume from here!")

# Get total remote size
head_resp = requests.head(URL, allow_redirects=True)
total_bytes = int(head_resp.headers.get("content-length", 0))
print(f"Remote file size: {total_bytes / (1024*1024):.1f} MB")

current_bytes = MODEL_FILE.stat().st_size if MODEL_FILE.exists() else 0
print(f"Current local size: {current_bytes / (1024*1024):.1f} MB ({current_bytes / total_bytes * 100:.1f}%)")

if current_bytes >= total_bytes:
    print("Download already complete!")
    sys.exit(0)

headers = {"Range": f"bytes={current_bytes}-"}
print(f"Resuming download from byte {current_bytes}...")

start_time = time.time()
last_report_time = start_time
last_report_bytes = current_bytes

with requests.get(URL, headers=headers, stream=True) as response:
    response.raise_for_status()
    with open(MODEL_FILE, "ab") as f:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024): # 4MB chunks
            if chunk:
                f.write(chunk)
                current_bytes += len(chunk)
                now = time.time()
                if now - last_report_time >= 3.0:
                    speed_mb = ((current_bytes - last_report_bytes) / (1024 * 1024)) / (now - last_report_time)
                    pct = (current_bytes / total_bytes) * 100
                    eta_sec = (total_bytes - current_bytes) / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                    print(f"Progress: {pct:.1f}% ({current_bytes / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB) | Speed: {speed_mb:.1f} MB/s | ETA: {int(eta_sec)}s")
                    last_report_time = now
                    last_report_bytes = current_bytes

print("Successfully finished downloading Hunyuan3D-2mv weights!")
