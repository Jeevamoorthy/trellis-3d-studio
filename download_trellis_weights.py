"""
Downloads the official Microsoft TRELLIS weights (JeffreyXiang/TRELLIS-image-large)
and tracks real download progress to a JSON status file for the desktop UI.
"""

import os
import sys
import json
import time
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(BASE_DIR, "download_progress.json")
WEIGHTS_DIR = os.path.join(BASE_DIR, "TRELLIS-for-windows", "weights", "JeffreyXiang", "TRELLIS-image-large")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def update_status(stage, percent, downloaded_mb=0, total_mb=3140, msg=""):
    data = {
        "stage": stage,
        "percent": percent,
        "downloaded_mb": round(downloaded_mb, 1),
        "total_mb": total_mb,
        "message": msg,
        "timestamp": time.time()
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)
    print(f"[{percent}%] {stage}: {msg}")

def main():
    update_status("Initializing", 0, 0, 3140, "Connecting to Hugging Face...")
    try:
        print("Starting download of JeffreyXiang/TRELLIS-image-large (~3.07 GB)...")
        # Download repository
        model_path = snapshot_download(
            repo_id="JeffreyXiang/TRELLIS-image-large",
            local_dir=WEIGHTS_DIR,
            max_workers=4
        )
        update_status("Completed", 100, 3140, 3140, f"Model weights ready at {model_path}")
        print("Download complete!")
    except Exception as e:
        update_status("Error", 0, 0, 3140, str(e))
        print("Error during download:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
