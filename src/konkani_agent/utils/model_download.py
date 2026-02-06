"""Model download utilities"""

import os
from pathlib import Path
import urllib.request
from loguru import logger

from config.settings import MODELS_DIR


# Model URLs
INDICCONFORMER_URL = (
    "https://objectstore.e2enetworks.net/indicconformer/models/"
    "indicconformer_stt_kok_hybrid_rnnt_large.nemo"
)


def download_indicconformer():
    """Download IndicConformer Konkani model."""
    model_path = MODELS_DIR / "indicconformer_stt_kok_hybrid_rnnt_large.nemo"

    if model_path.exists():
        logger.info(f"✓ Model already exists: {model_path}")
        return str(model_path)

    logger.info("Downloading IndicConformer Konkani model...")
    logger.info(f"  URL: {INDICCONFORMER_URL}")
    logger.info(f"  Size: ~499MB")
    logger.info(f"  Destination: {model_path}")

    try:
        # Create models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            if block_num % 100 == 0:  # Log every 100 blocks
                logger.info(f"  Downloaded: {percent:.1f}%")

        urllib.request.urlretrieve(
            INDICCONFORMER_URL, model_path, reporthook=report_progress
        )

        logger.info(f"✓ Model downloaded: {model_path}")
        return str(model_path)

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def verify_model():
    """Verify model file exists and has correct size."""
    model_path = MODELS_DIR / "indicconformer_stt_kok_hybrid_rnnt_large.nemo"

    if not model_path.exists():
        logger.error(f"✗ Model not found: {model_path}")
        return False

    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Model found: {model_path} ({size_mb:.1f} MB)")

    # Expected size is around 499MB
    if size_mb < 400:
        logger.warning(f"⚠ Model size seems too small ({size_mb:.1f} MB)")
        return False

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        success = verify_model()
        sys.exit(0 if success else 1)
    else:
        download_indicconformer()
