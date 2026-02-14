#!/usr/bin/env python3
"""
Quick model downloader - run this to download the STT model
"""

import urllib.request
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "indicconformer_stt_kok_hybrid_rnnt_large.nemo"

if model_path.exists():
    print(f"✓ Model already exists: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024**2):.1f} MB")
else:
    url = "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_kok_hybrid_rnnt_large.nemo"
    print(f"Downloading Konkani STT model...")
    print(f"Source: {url}")
    print("Size: 499 MB (this will take a few minutes)")
    print()

    try:

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100
            print(
                f"\rProgress: {percent:.1f}% ({downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB)",
                end="",
            )

        urllib.request.urlretrieve(url, model_path, progress)
        print()
        print(f"\n✓ Download complete!")
        print(f"  Location: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024**2):.1f} MB")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
