#!/usr/bin/env python3
"""
Main entry point for the Stable Diffusion ControlNet Pipeline
for Custom Print Artwork Generation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from src.generators.controlnet_pipeline import ControlNetPipeline
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_output_dir

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion ControlNet Pipeline for Custom Print Artwork"
    )

    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for image generation"
    )

    parser.add_argument(
        "--style",
        type=str,
        default="graphic",
        choices=["photorealistic", "illustrative", "graphic", "minimalist", "tattoo"],
        help="Art style for generation",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["mockup", "flattened", "both"],
        help="Output format: mockup, flattened design, or both",
    )

    parser.add_argument(
        "--variations",
        type=int,
        default=1,
        help="Number of variations to generate (1-4)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated images",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible generation"
    )

    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="Generation quality setting",
    )

    return parser.parse_args()


def generate_single_design(
    pipeline: ControlNetPipeline,
    prompt: str,
    style: str,
    output_format: str,
    output_dir: Path,
    seed: Optional[int] = None,
) -> dict:
    """Generate a single design with the specified parameters."""

    logger.info(f"Generating design with prompt: '{prompt}'")
    logger.info(f"Style: {style}, Format: {output_format}")

    try:
        result = pipeline.generate(
            prompt=prompt, style=style, output_format=output_format, seed=seed
        )

        # Save outputs
        output_files = {}
        if "mockup" in result:
            mockup_path = output_dir / f"mockup_{style}_{seed or 'random'}.png"
            result["mockup"].save(mockup_path)
            output_files["mockup"] = str(mockup_path)
            logger.info(f"Mockup saved to: {mockup_path}")

        if "flattened" in result:
            flattened_path = output_dir / f"flattened_{style}_{seed or 'random'}.png"
            result["flattened"].save(flattened_path)
            output_files["flattened"] = str(flattened_path)
            logger.info(f"Flattened design saved to: {flattened_path}")

        return {
            "success": True,
            "output_files": output_files,
            "metadata": {
                "prompt": prompt,
                "style": style,
                "seed": seed,
                "output_format": output_format,
            },
        }

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "metadata": {"prompt": prompt, "style": style, "seed": seed},
        }


def generate_variations(
    pipeline: ControlNetPipeline,
    prompt: str,
    style: str,
    num_variations: int,
    output_dir: Path,
) -> dict:
    """Generate multiple variations of a design."""

    logger.info(f"Generating {num_variations} variations for prompt: '{prompt}'")

    variations = []
    for i in range(num_variations):
        logger.info(f"Generating variation {i+1}/{num_variations}")

        result = generate_single_design(
            pipeline=pipeline,
            prompt=prompt,
            style=style,
            output_format="both",
            output_dir=output_dir / f"variation_{i+1}",
            seed=None,  # Random seed for variations
        )

        variations.append(result)

    # Create preview grid if multiple variations
    if num_variations > 1:
        try:
            grid_path = pipeline.create_preview_grid(
                variations, output_dir / "preview_grid.png"
            )
            logger.info(f"Preview grid saved to: {grid_path}")
        except Exception as e:
            logger.warning(f"Failed to create preview grid: {str(e)}")

    return {
        "success": True,
        "variations": variations,
        "total_generated": len(variations),
    }


def main():
    """Main application entry point."""

    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    # Initialize pipeline
    try:
        pipeline = ControlNetPipeline(config=config)
        logger.info("ControlNet pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        sys.exit(1)

    # Generate designs
    if args.variations == 1:
        result = generate_single_design(
            pipeline=pipeline,
            prompt=args.prompt,
            style=args.style,
            output_format=args.output_format,
            output_dir=output_dir,
            seed=args.seed,
        )
    else:
        result = generate_variations(
            pipeline=pipeline,
            prompt=args.prompt,
            style=args.style,
            num_variations=min(args.variations, 4),  # Max 4 variations
            output_dir=output_dir,
        )

    # Report results
    if result.get("success", False):
        logger.info("Generation completed successfully!")
        if "output_files" in result:
            logger.info(f"Output files: {result['output_files']}")
        elif "total_generated" in result:
            logger.info(f"Generated {result['total_generated']} variations")
    else:
        logger.error(f"Generation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
