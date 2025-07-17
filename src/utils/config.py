"""
Configuration management utilities for the pipeline
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ["models", "generation", "output"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration settings.

    Returns:
        Dictionary with default configuration
    """
    return {
        "models": {
            "stable_diffusion": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "revision": "fp16",
                "torch_dtype": "float16",
            },
            "controlnet": {
                "model_id": "lllyasviel/control_v11p_sd15_canny",
                "revision": "fp16",
                "torch_dtype": "float16",
            },
        },
        "generation": {
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "num_images_per_prompt": 1,
        },
        "output": {
            "format": "PNG",
            "dpi": 300,
            "band_width_mm": 134,
            "band_height_mm": 25,
            "sphere_diameter_mm": 42.67,
        },
        "styles": {
            "photorealistic": {
                "prompt_suffix": ", highly detailed, photorealistic, 8k resolution",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing",
            },
            "illustrative": {
                "prompt_suffix": ", artistic illustration, hand-drawn style",
                "negative_prompt": "photorealistic, photograph, 3d render",
            },
            "graphic": {
                "prompt_suffix": ", bold graphic design, modern, clean",
                "negative_prompt": "photorealistic, illustration, painting",
            },
            "minimalist": {
                "prompt_suffix": ", minimalist design, simple, clean, elegant",
                "negative_prompt": "complex, detailed, busy, cluttered",
            },
            "tattoo": {
                "prompt_suffix": ", tattoo art style, intricate details, black and white",
                "negative_prompt": "colorful, simple, minimalist",
            },
        },
    }
