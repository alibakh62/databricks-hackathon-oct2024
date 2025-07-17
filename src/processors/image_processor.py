"""
Image processing utilities for the pipeline
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageProcessor:
    """
    Handles image processing operations for the pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_config = config["output"]

        # Calculate band dimensions in pixels at 300 DPI
        self.band_width_px = int(self.output_config["band_width_mm"] * 300 / 25.4)
        self.band_height_px = int(self.output_config["band_height_mm"] * 300 / 25.4)

        logger.info(
            f"Band dimensions: {self.band_width_px}x{self.band_height_px} pixels"
        )

    def extract_band(self, image: Image.Image) -> Image.Image:
        """
        Extract the flattened band from a generated image.

        Args:
            image: Generated image containing the design

        Returns:
            Flattened band image ready for printing
        """
        # Convert to numpy array for processing
        img_array = np.array(image)

        # Find the band region (center horizontal band)
        height, width = img_array.shape[:2]
        band_height = height // 4
        band_y = (height - band_height) // 2

        # Extract band region
        band_region = img_array[band_y : band_y + band_height, :, :]

        # Resize to target dimensions
        band_image = Image.fromarray(band_region)
        band_image = band_image.resize(
            (self.band_width_px, self.band_height_px), Image.Resampling.LANCZOS
        )

        # Apply post-processing for print quality
        band_image = self._apply_print_optimization(band_image)

        return band_image

    def _apply_print_optimization(self, image: Image.Image) -> Image.Image:
        """
        Apply optimizations for print quality.

        Args:
            image: Input image

        Returns:
            Optimized image for printing
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply slight sharpening for crisp edges
        image = image.filter(
            ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
        )

        # Ensure proper contrast
        image = self._adjust_contrast(image, factor=1.1)

        return image

    def _adjust_contrast(self, image: Image.Image, factor: float = 1.0) -> Image.Image:
        """
        Adjust image contrast.

        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)

        Returns:
            Image with adjusted contrast
        """
        if factor == 1.0:
            return image

        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)

        # Apply contrast adjustment
        img_array = (img_array - 128) * factor + 128

        # Clip values to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def create_band_mask(self, width: int, height: int) -> Image.Image:
        """
        Create a mask for the band region.

        Args:
            width: Image width
            height: Image height

        Returns:
            Band mask image
        """
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Calculate band position
        band_height = height // 4
        band_y = (height - band_height) // 2

        # Draw band region
        draw.rectangle([0, band_y, width, band_y + band_height], fill=255)

        return mask

    def apply_band_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Apply band mask to image.

        Args:
            image: Input image
            mask: Band mask

        Returns:
            Masked image
        """
        # Ensure same size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)

        # Apply mask
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.paste(image, mask=mask)

        return result

    def validate_print_ready(self, image: Image.Image) -> bool:
        """
        Validate if image is ready for printing.

        Args:
            image: Image to validate

        Returns:
            True if image is print-ready
        """
        # Check dimensions
        if image.size != (self.band_width_px, self.band_height_px):
            logger.warning(
                f"Image dimensions {image.size} don't match expected {self.band_width_px}x{self.band_height_px}"
            )
            return False

        # Check mode
        if image.mode not in ["RGB", "RGBA"]:
            logger.warning(f"Image mode {image.mode} not suitable for printing")
            return False

        # Check for empty regions
        img_array = np.array(image)
        if img_array.mean() < 10:  # Very dark image
            logger.warning("Image appears too dark for printing")
            return False

        return True

    def convert_to_cmyk(self, image: Image.Image) -> Image.Image:
        """
        Convert RGB image to CMYK for printing.

        Args:
            image: RGB image

        Returns:
            CMYK image
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to CMYK
        cmyk_image = image.convert("CMYK")

        return cmyk_image

    def add_print_margins(self, image: Image.Image, margin_px: int = 10) -> Image.Image:
        """
        Add print margins to the image.

        Args:
            image: Input image
            margin_px: Margin size in pixels

        Returns:
            Image with margins
        """
        width, height = image.size
        new_width = width + 2 * margin_px
        new_height = height + 2 * margin_px

        # Create new image with white background
        result = Image.new("RGB", (new_width, new_height), (255, 255, 255))

        # Paste original image in center
        result.paste(image, (margin_px, margin_px))

        return result

    def create_preview_thumbnail(
        self, image: Image.Image, size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """
        Create a thumbnail for preview.

        Args:
            image: Input image
            size: Thumbnail size

        Returns:
            Thumbnail image
        """
        return image.resize(size, Image.Resampling.LANCZOS)
