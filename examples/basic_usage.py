#!/usr/bin/env python3
"""
Basic usage example for the Stable Diffusion ControlNet Pipeline
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_default_config
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_output_dir

logger = setup_logger(__name__)


def example_basic_generation():
    """Example of basic image generation without actual models."""
    logger.info("Running basic generation example...")

    # Load configuration
    config = get_default_config()

    # Create output directory
    output_dir = Path("outputs/example")
    ensure_output_dir(output_dir)

    # Example prompts for different styles
    examples = [
        {
            "prompt": "BEAST MODE in heavy blackletter font with lightning bolts",
            "style": "graphic",
            "description": "Bold graphic design with lightning",
        },
        {
            "prompt": "Minimalist geometric pattern with clean lines",
            "style": "minimalist",
            "description": "Clean minimalist design",
        },
        {
            "prompt": "Traditional tattoo design with skull and roses",
            "style": "tattoo",
            "description": "Tattoo-style artwork",
        },
        {
            "prompt": "Photorealistic mountain landscape at sunset",
            "style": "photorealistic",
            "description": "Realistic landscape",
        },
    ]

    logger.info("Example prompts configured:")
    for i, example in enumerate(examples, 1):
        logger.info(f"  {i}. {example['description']}")
        logger.info(f"     Prompt: '{example['prompt']}'")
        logger.info(f"     Style: {example['style']}")

    logger.info("\nTo run actual generation, install dependencies and run:")
    logger.info("  pip install -r requirements.txt")
    logger.info("  python scripts/download_models.py")
    logger.info("  python main.py --prompt 'Your prompt here' --style graphic")


def example_configuration():
    """Example of configuration customization."""
    logger.info("Configuration example:")

    config = get_default_config()

    # Show current settings
    logger.info(f"Generation settings:")
    logger.info(f"  Steps: {config['generation']['num_inference_steps']}")
    logger.info(f"  Guidance scale: {config['generation']['guidance_scale']}")
    logger.info(
        f"  Image size: {config['generation']['width']}x{config['generation']['height']}"
    )

    logger.info(f"\nOutput settings:")
    logger.info(
        f"  Band dimensions: {config['output']['band_width_mm']}x{config['output']['band_height_mm']}mm"
    )
    logger.info(f"  DPI: {config['output']['dpi']}")
    logger.info(f"  Sphere diameter: {config['output']['sphere_diameter_mm']}mm")

    logger.info(f"\nAvailable styles:")
    for style in config["styles"].keys():
        logger.info(f"  - {style}")


def example_pipeline_workflow():
    """Example of the complete pipeline workflow."""
    logger.info("Pipeline workflow example:")

    workflow_steps = [
        "1. Load configuration and initialize pipeline",
        "2. Download/load Stable Diffusion and ControlNet models",
        "3. Create control image for band layout",
        "4. Generate base image using Stable Diffusion + ControlNet",
        "5. Extract flattened band for printing",
        "6. Create product mockup with 3D sphere",
        "7. Save outputs (mockup + flattened design)",
        "8. Optionally generate variations and preview grid",
    ]

    for step in workflow_steps:
        logger.info(f"  {step}")

    logger.info("\nExpected outputs:")
    logger.info("  - Product mockup (800x600 PNG)")
    logger.info("  - Flattened design (1583x118 pixels @ 300 DPI)")
    logger.info("  - Preview grid (2x2 variations if requested)")


if __name__ == "__main__":
    logger.info("=== Stable Diffusion ControlNet Pipeline Examples ===\n")

    example_basic_generation()
    print()

    example_configuration()
    print()

    example_pipeline_workflow()
    print()

    logger.info("Examples completed. Check the documentation for more details.")
