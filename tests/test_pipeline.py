"""
Unit tests for the ControlNet pipeline
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_default_config
from src.processors.image_processor import ImageProcessor
from src.mockups.product_mockup import ProductMockupGenerator


class TestImageProcessor(unittest.TestCase):
    """Test cases for ImageProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.processor = ImageProcessor(self.config)

    def test_band_dimensions_calculation(self):
        """Test band dimensions calculation."""
        expected_width = int(134 * 300 / 25.4)  # 134mm at 300 DPI
        expected_height = int(25 * 300 / 25.4)  # 25mm at 300 DPI

        self.assertEqual(self.processor.band_width_px, expected_width)
        self.assertEqual(self.processor.band_height_px, expected_height)

    def test_create_band_mask(self):
        """Test band mask creation."""
        width, height = 512, 512
        mask = self.processor.create_band_mask(width, height)

        self.assertEqual(mask.size, (width, height))
        self.assertEqual(mask.mode, "L")

        # Check that mask has some white pixels (band region)
        mask_array = np.array(mask)
        self.assertTrue(np.any(mask_array > 0))

    def test_adjust_contrast(self):
        """Test contrast adjustment."""
        # Create test image
        test_image = Image.new("RGB", (100, 100), (128, 128, 128))

        # Test contrast increase
        adjusted = self.processor._adjust_contrast(test_image, factor=1.5)
        adjusted_array = np.array(adjusted)

        # Should have different values after contrast adjustment
        self.assertFalse(np.array_equal(np.array(test_image), adjusted_array))

    def test_validate_print_ready(self):
        """Test print-ready validation."""
        # Create valid image
        valid_image = Image.new(
            "RGB",
            (self.processor.band_width_px, self.processor.band_height_px),
            (255, 255, 255),
        )
        self.assertTrue(self.processor.validate_print_ready(valid_image))

        # Create invalid image (wrong size)
        invalid_image = Image.new("RGB", (100, 100), (255, 255, 255))
        self.assertFalse(self.processor.validate_print_ready(invalid_image))


class TestProductMockupGenerator(unittest.TestCase):
    """Test cases for ProductMockupGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.generator = ProductMockupGenerator(self.config)

    def test_initialization(self):
        """Test mockup generator initialization."""
        self.assertEqual(self.generator.sphere_diameter_mm, 42.67)
        self.assertEqual(self.generator.band_width_mm, 134)
        self.assertEqual(self.generator.band_height_mm, 25)
        self.assertEqual(self.generator.mockup_width, 800)
        self.assertEqual(self.generator.mockup_height, 600)

    def test_create_background(self):
        """Test background creation."""
        background = self.generator._create_background()

        self.assertEqual(background.size, (800, 600))
        self.assertEqual(background.mode, "RGB")

    def test_create_sphere(self):
        """Test sphere creation."""
        sphere = self.generator._create_sphere()

        # Should be square
        self.assertEqual(sphere.size[0], sphere.size[1])
        self.assertEqual(sphere.mode, "RGBA")

        # Should have some transparent pixels
        sphere_array = np.array(sphere)
        self.assertTrue(np.any(sphere_array[:, :, 3] == 0))  # Transparent pixels

    def test_extract_band_from_design(self):
        """Test band extraction from design."""
        # Create test design image
        design = Image.new("RGB", (512, 512), (255, 0, 0))  # Red image

        band = self.generator._extract_band_from_design(design)

        # Should be smaller than original
        self.assertLess(band.size[1], design.size[1])
        self.assertEqual(band.size[0], design.size[0])


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management."""

    def test_default_config_structure(self):
        """Test default configuration structure."""
        config = get_default_config()

        # Check required sections exist
        self.assertIn("models", config)
        self.assertIn("generation", config)
        self.assertIn("output", config)
        self.assertIn("styles", config)

        # Check model configurations
        self.assertIn("stable_diffusion", config["models"])
        self.assertIn("controlnet", config["models"])

        # Check style configurations
        expected_styles = [
            "photorealistic",
            "illustrative",
            "graphic",
            "minimalist",
            "tattoo",
        ]
        for style in expected_styles:
            self.assertIn(style, config["styles"])


if __name__ == "__main__":
    unittest.main()
