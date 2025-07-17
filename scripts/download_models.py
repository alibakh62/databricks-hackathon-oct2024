#!/usr/bin/env python3
"""
Script to download required models for the Stable Diffusion ControlNet Pipeline
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_default_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def download_model(model_id: str, model_type: str, revision: str = "fp16") -> bool:
    """
    Download a model from Hugging Face.

    Args:
        model_id: Hugging Face model ID
        model_type: Type of model (stable_diffusion, controlnet)
        revision: Model revision to download

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading {model_type} model: {model_id}")

        if model_type == "stable_diffusion":
            from diffusers import StableDiffusionPipeline

            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, revision=revision, torch_dtype="auto"
            )
            logger.info(f"Successfully downloaded Stable Diffusion model: {model_id}")

        elif model_type == "controlnet":
            from diffusers import ControlNetModel

            model = ControlNetModel.from_pretrained(
                model_id, revision=revision, torch_dtype="auto"
            )
            logger.info(f"Successfully downloaded ControlNet model: {model_id}")

        else:
            logger.error(f"Unknown model type: {model_type}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to download {model_type} model {model_id}: {str(e)}")
        return False


def download_all_models(config: Dict[str, Any]) -> bool:
    """
    Download all required models from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if all downloads successful, False otherwise
    """
    models_config = config.get("models", {})

    success = True

    # Download Stable Diffusion model
    if "stable_diffusion" in models_config:
        sd_config = models_config["stable_diffusion"]
        model_id = sd_config.get("model_id")
        revision = sd_config.get("revision", "fp16")

        if model_id:
            if not download_model(model_id, "stable_diffusion", revision):
                success = False
        else:
            logger.error("Stable Diffusion model ID not found in configuration")
            success = False

    # Download ControlNet model
    if "controlnet" in models_config:
        cn_config = models_config["controlnet"]
        model_id = cn_config.get("model_id")
        revision = cn_config.get("revision", "fp16")

        if model_id:
            if not download_model(model_id, "controlnet", revision):
                success = False
        else:
            logger.error("ControlNet model ID not found in configuration")
            success = False

    return success


def check_model_cache() -> Dict[str, bool]:
    """
    Check which models are already cached.

    Returns:
        Dictionary mapping model types to availability status
    """
    from huggingface_hub import snapshot_download
    from diffusers import StableDiffusionPipeline, ControlNetModel

    cache_status = {}

    try:
        # Check Stable Diffusion model
        try:
            StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", revision="fp16", local_files_only=True
            )
            cache_status["stable_diffusion"] = True
            logger.info("Stable Diffusion model found in cache")
        except:
            cache_status["stable_diffusion"] = False
            logger.info("Stable Diffusion model not found in cache")

        # Check ControlNet model
        try:
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                revision="fp16",
                local_files_only=True,
            )
            cache_status["controlnet"] = True
            logger.info("ControlNet model found in cache")
        except:
            cache_status["controlnet"] = False
            logger.info("ControlNet model not found in cache")

    except Exception as e:
        logger.error(f"Error checking model cache: {str(e)}")
        cache_status = {"stable_diffusion": False, "controlnet": False}

    return cache_status


def main():
    """Main function for model download script."""
    parser = argparse.ArgumentParser(
        description="Download models for Stable Diffusion ControlNet Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which models are cached, don't download",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if models are already cached",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        if Path(args.config).exists():
            config = load_config(args.config)
            logger.info(f"Configuration loaded from: {args.config}")
        else:
            logger.warning(f"Configuration file not found: {args.config}")
            logger.info("Using default configuration")
            config = get_default_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

    # Check cache status
    cache_status = check_model_cache()

    if args.check_only:
        logger.info("Cache check completed")
        for model_type, cached in cache_status.items():
            status = "✓ Cached" if cached else "✗ Not cached"
            logger.info(f"{model_type}: {status}")
        return

    # Download models if needed
    if not args.force:
        all_cached = all(cache_status.values())
        if all_cached:
            logger.info("All models are already cached. Use --force to re-download.")
            return

    # Download models
    logger.info("Starting model downloads...")
    success = download_all_models(config)

    if success:
        logger.info("All models downloaded successfully!")
    else:
        logger.error("Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()
