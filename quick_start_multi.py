#!/usr/bin/env python3
"""
Quick downloader for IndicConformer 600M (Multilingual)
No huge libraries required. Just Python.
"""

import urllib.request
from pathlib import Path
import sys

# 1. Setup paths
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

filename = "indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
model_path = models_dir / filename

# 2. The Direct Link (Based on AI4Bharat ObjectStore convention)
url = f"https://objectstore.e2enetworks.net/indicconformer/models/{filename}"

if model_path.exists():
    print(f"✓ Model already exists: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024**2):.1f} MB")
    sys.exit(0)

print(f"Downloading 600M Multilingual STT model...")
print(f"Source: {url}")
print("Size: ~2.4 GB (This will take a while!)")
print()

try:
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            # Print progress bar
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, model_path, progress)
    
    print()
    print(f"\n✓ Download complete!")
    print(f"  Location: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024**2):.1f} MB")

except Exception as e:
    print(f"\n\n✗ Download failed: {e}")
    print("Tip: If this link fails, AI4Bharat might have moved it.")
    print("Fallback: You can use 'huggingface-cli download ai4bharat/indic-conformer-600m-multilingual' if you install the library.")
    # Cleanup partial file
    if model_path.exists():
        model_path.unlink()