
import os
import requests
import shutil

def download_video(url: str, dest_path: str):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(response.raw, f)
    print(f"✅ Downloaded video to {dest_path}")

def clean_folder_if_needed(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"🧹 Cleaned folder {folder}")
