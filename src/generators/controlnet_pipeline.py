"""
Main ControlNet pipeline for generating custom print artwork
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

from src.processors.image_processor import ImageProcessor
from src.mockups.product_mockup import ProductMockupGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ControlNetPipeline:
    """
    Main pipeline for generating custom print artwork using Stable Diffusion and ControlNet.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ControlNet pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self._load_models()
        self.image_processor = ImageProcessor(config)
        self.mockup_generator = ProductMockupGenerator(config)

        logger.info("ControlNet pipeline initialized successfully")

    def _load_models(self):
        """Load Stable Diffusion and ControlNet models."""
        try:
            # Load ControlNet model
            controlnet_model_id = self.config["models"]["controlnet"]["model_id"]
            logger.info(f"Loading ControlNet model: {controlnet_model_id}")

            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                revision=self.config["models"]["controlnet"]["revision"],
            )

            # Load Stable Diffusion pipeline
            sd_model_id = self.config["models"]["stable_diffusion"]["model_id"]
            logger.info(f"Loading Stable Diffusion model: {sd_model_id}")

            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                sd_model_id,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                revision=self.config["models"]["stable_diffusion"]["revision"],
            )

            # Move to device
            self.pipeline.to(self.device)

            # Use faster scheduler
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        style: str = "graphic",
        output_format: str = "both",
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Image.Image]:
        """
        Generate artwork based on prompt and style.

        Args:
            prompt: Text prompt for generation
            style: Art style (photorealistic, illustrative, graphic, minimalist, tattoo)
            output_format: Output format (mockup, flattened, both)
            seed: Random seed for reproducible generation
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated images
        """
        logger.info(f"Generating artwork with prompt: '{prompt}'")
        logger.info(f"Style: {style}, Format: {output_format}")

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare prompt with style
        enhanced_prompt = self._enhance_prompt(prompt, style)
        negative_prompt = self._get_negative_prompt(style)

        # Generate control image (band layout)
        control_image = self._create_control_image()

        # Generate base image
        base_image = self._generate_base_image(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            **kwargs,
        )

        # Process results
        result = {}

        if output_format in ["mockup", "both"]:
            mockup = self.mockup_generator.create_mockup(base_image)
            result["mockup"] = mockup

        if output_format in ["flattened", "both"]:
            flattened = self.image_processor.extract_band(base_image)
            result["flattened"] = flattened

        logger.info("Artwork generation completed")
        return result

    def generate_variations(
        self, prompt: str, style: str = "graphic", num_variations: int = 4, **kwargs
    ) -> List[Dict[str, Image.Image]]:
        """
        Generate multiple variations of a design.

        Args:
            prompt: Text prompt for generation
            style: Art style
            num_variations: Number of variations to generate (1-4)
            **kwargs: Additional generation parameters

        Returns:
            List of variation results
        """
        logger.info(f"Generating {num_variations} variations")

        variations = []
        for i in range(num_variations):
            logger.info(f"Generating variation {i+1}/{num_variations}")

            # Use different seed for each variation
            seed = np.random.randint(0, 2**32) if i > 0 else None

            variation = self.generate(
                prompt=prompt, style=style, output_format="both", seed=seed, **kwargs
            )

            variations.append(variation)

        return variations

    def create_preview_grid(
        self, variations: List[Dict[str, Image.Image]], output_path: Union[str, Path]
    ) -> Path:
        """
        Create a 2x2 preview grid of variations.

        Args:
            variations: List of variation results
            output_path: Path where to save the grid

        Returns:
            Path to saved grid image
        """
        from src.utils.file_utils import create_preview_grid

        # Extract mockup images for grid
        mockup_images = []
        for variation in variations[:4]:  # Max 4 variations
            if "mockup" in variation:
                mockup_images.append(variation["mockup"])

        # Create grid
        grid_path = create_preview_grid(
            images=mockup_images, output_path=output_path, grid_size=(2, 2)
        )

        return grid_path

    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """
        Enhance prompt with style-specific modifiers.

        Args:
            prompt: Original prompt
            style: Art style

        Returns:
            Enhanced prompt
        """
        style_config = self.config["styles"].get(style, {})
        suffix = style_config.get("prompt_suffix", "")

        enhanced = f"{prompt}{suffix}"

        # Add band-specific context
        enhanced += ", designed for a spherical band, centered composition"

        return enhanced

    def _get_negative_prompt(self, style: str) -> str:
        """
        Get negative prompt for the specified style.

        Args:
            style: Art style

        Returns:
            Negative prompt
        """
        style_config = self.config["styles"].get(style, {})
        negative = style_config.get("negative_prompt", "")

        # Add general negative prompts
        negative += ", blurry, low quality, distorted, off-center"

        return negative

    def _create_control_image(self) -> Image.Image:
        """
        Create control image for the band layout.

        Returns:
            Control image for ControlNet
        """
        # Create a simple band layout control image
        width = self.config["generation"]["width"]
        height = self.config["generation"]["height"]

        # Create band mask (center horizontal band)
        control_image = Image.new("RGB", (width, height), (0, 0, 0))

        # Calculate band position (center 25% of height)
        band_height = height // 4
        band_y = (height - band_height) // 2

        # Draw band outline
        from PIL import ImageDraw

        draw = ImageDraw.Draw(control_image)
        draw.rectangle(
            [0, band_y, width, band_y + band_height], outline=(255, 255, 255), width=2
        )

        return control_image

    def _generate_base_image(
        self, prompt: str, negative_prompt: str, control_image: Image.Image, **kwargs
    ) -> Image.Image:
        """
        Generate base image using Stable Diffusion with ControlNet.

        Args:
            prompt: Enhanced prompt
            negative_prompt: Negative prompt
            control_image: Control image for layout
            **kwargs: Additional generation parameters

        Returns:
            Generated base image
        """
        # Get generation parameters
        gen_config = self.config["generation"]
        num_steps = kwargs.get("num_inference_steps", gen_config["num_inference_steps"])
        guidance_scale = kwargs.get("guidance_scale", gen_config["guidance_scale"])
        width = kwargs.get("width", gen_config["width"])
        height = kwargs.get("height", gen_config["height"])

        logger.info(f"Generating image: {width}x{height}, {num_steps} steps")

        # Generate image
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=1,
            )

        # Extract generated image
        generated_image = result.images[0]

        return generated_image

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, "pipeline"):
            del self.pipeline
        if hasattr(self, "controlnet"):
            del self.controlnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
