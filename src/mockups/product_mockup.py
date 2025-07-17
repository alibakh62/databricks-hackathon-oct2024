"""
Product mockup generation for spherical objects
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, Tuple, Optional
import math

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProductMockupGenerator:
    """
    Generates product mockups by applying designs to 3D spherical objects.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mockup generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_config = config["output"]

        # Sphere parameters
        self.sphere_diameter_mm = self.output_config["sphere_diameter_mm"]
        self.band_width_mm = self.output_config["band_width_mm"]
        self.band_height_mm = self.output_config["band_height_mm"]

        # Mockup image dimensions
        self.mockup_width = 800
        self.mockup_height = 600

        logger.info(
            f"Mockup generator initialized for {self.sphere_diameter_mm}mm sphere"
        )

    def create_mockup(self, design_image: Image.Image) -> Image.Image:
        """
        Create a product mockup by applying the design to a spherical object.

        Args:
            design_image: Generated design image

        Returns:
            Product mockup image
        """
        # Create base mockup background
        mockup = self._create_background()

        # Create spherical object
        sphere = self._create_sphere()

        # Apply design to sphere
        sphere_with_design = self._apply_design_to_sphere(sphere, design_image)

        # Composite sphere onto mockup
        result = self._composite_sphere(mockup, sphere_with_design)

        # Add lighting and shadows
        result = self._add_lighting_and_shadows(result, sphere_with_design)

        return result

    def _create_background(self) -> Image.Image:
        """
        Create a background for the mockup.

        Returns:
            Background image
        """
        # Create gradient background
        background = Image.new("RGB", (self.mockup_width, self.mockup_height))
        draw = ImageDraw.Draw(background)

        # Create gradient from top to bottom
        for y in range(self.mockup_height):
            # Calculate gradient color (light gray to white)
            ratio = y / self.mockup_height
            color_value = int(240 + ratio * 15)  # 240-255 range
            color = (color_value, color_value, color_value)

            draw.line([(0, y), (self.mockup_width, y)], fill=color)

        return background

    def _create_sphere(self) -> Image.Image:
        """
        Create a basic spherical shape.

        Returns:
            Sphere image with alpha channel
        """
        # Sphere size in pixels (scale based on mockup size)
        sphere_radius = min(self.mockup_width, self.mockup_height) // 3

        # Create sphere with alpha channel
        sphere_size = sphere_radius * 2
        sphere = Image.new("RGBA", (sphere_size, sphere_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sphere)

        # Draw sphere with gradient for 3D effect
        center_x, center_y = sphere_radius, sphere_radius

        for y in range(sphere_size):
            for x in range(sphere_size):
                # Calculate distance from center
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance <= sphere_radius:
                    # Calculate alpha based on distance (fade at edges)
                    alpha = int(255 * (1 - distance / sphere_radius))

                    # Calculate lighting effect (simulate 3D lighting)
                    # Light source from top-left
                    light_x = (x - center_x) / sphere_radius
                    light_y = (y - center_y) / sphere_radius

                    # Simple lighting calculation
                    lighting = 0.7 + 0.3 * (1 - light_x - light_y) / 2
                    lighting = max(0.3, min(1.0, lighting))

                    # Base sphere color (light gray)
                    base_color = int(200 * lighting)
                    color = (base_color, base_color, base_color, alpha)

                    sphere.putpixel((x, y), color)

        return sphere

    def _apply_design_to_sphere(
        self, sphere: Image.Image, design_image: Image.Image
    ) -> Image.Image:
        """
        Apply the design to the spherical surface.

        Args:
            sphere: Base sphere image
            design_image: Design to apply

        Returns:
            Sphere with applied design
        """
        # Extract band region from design
        band_image = self._extract_band_from_design(design_image)

        # Create spherical mapping
        sphere_with_design = sphere.copy()
        sphere_array = np.array(sphere_with_design)

        # Get sphere dimensions
        sphere_width, sphere_height = sphere.size
        center_x, center_y = sphere_width // 2, sphere_height // 2
        radius = min(center_x, center_y)

        # Convert band image to array
        band_array = np.array(band_image)
        band_height, band_width = band_array.shape[:2]

        # Apply design to sphere using spherical mapping
        for y in range(sphere_height):
            for x in range(sphere_width):
                # Calculate spherical coordinates
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance <= radius:
                    # Calculate latitude and longitude
                    latitude = math.asin(dy / radius)  # -π/2 to π/2
                    longitude = math.atan2(
                        dx, math.sqrt(radius * radius - dy * dy)
                    )  # -π to π

                    # Map to band coordinates
                    # Longitude maps to band width (wraps around)
                    band_x = int((longitude + math.pi) / (2 * math.pi) * band_width)
                    band_x = band_x % band_width

                    # Latitude maps to band height (only center portion)
                    band_y = int((latitude + math.pi / 4) / (math.pi / 2) * band_height)
                    band_y = max(0, min(band_height - 1, band_y))

                    # Get design color
                    if 0 <= band_x < band_width and 0 <= band_y < band_height:
                        design_color = band_array[band_y, band_x]

                        # Blend with sphere color
                        sphere_color = sphere_array[y, x]
                        if sphere_color[3] > 0:  # If sphere pixel is not transparent
                            # Blend design with sphere lighting
                            alpha = sphere_color[3] / 255.0
                            blended_color = tuple(
                                int(design_color[i] * 0.7 + sphere_color[i] * 0.3)
                                for i in range(3)
                            )
                            sphere_array[y, x] = (*blended_color, sphere_color[3])

        return Image.fromarray(sphere_array)

    def _extract_band_from_design(self, design_image: Image.Image) -> Image.Image:
        """
        Extract the band region from the design image.

        Args:
            design_image: Full design image

        Returns:
            Band region image
        """
        # Convert to array
        img_array = np.array(design_image)
        height, width = img_array.shape[:2]

        # Extract center band (25% of height)
        band_height = height // 4
        band_y = (height - band_height) // 2

        band_region = img_array[band_y : band_y + band_height, :, :]

        return Image.fromarray(band_region)

    def _composite_sphere(
        self, background: Image.Image, sphere: Image.Image
    ) -> Image.Image:
        """
        Composite sphere onto background.

        Args:
            background: Background image
            sphere: Sphere with design

        Returns:
            Composited mockup
        """
        # Calculate sphere position (center of background)
        bg_width, bg_height = background.size
        sphere_width, sphere_height = sphere.size

        x = (bg_width - sphere_width) // 2
        y = (bg_height - sphere_height) // 2

        # Create result image
        result = background.copy()

        # Composite sphere onto background
        result.paste(sphere, (x, y), sphere)

        return result

    def _add_lighting_and_shadows(
        self, mockup: Image.Image, sphere: Image.Image
    ) -> Image.Image:
        """
        Add lighting effects and shadows to the mockup.

        Args:
            mockup: Base mockup image
            sphere: Sphere image

        Returns:
            Mockup with lighting and shadows
        """
        # Create shadow
        shadow = self._create_shadow(sphere)

        # Composite shadow first
        sphere_width, sphere_height = sphere.size
        bg_width, bg_height = mockup.size

        shadow_x = (bg_width - sphere_width) // 2 + 10  # Offset shadow
        shadow_y = (bg_height - sphere_height) // 2 + 10

        result = mockup.copy()
        result.paste(shadow, (shadow_x, shadow_y), shadow)

        # Re-composite sphere on top
        sphere_x = (bg_width - sphere_width) // 2
        sphere_y = (bg_height - sphere_height) // 2
        result.paste(sphere, (sphere_x, sphere_y), sphere)

        return result

    def _create_shadow(self, sphere: Image.Image) -> Image.Image:
        """
        Create a shadow for the sphere.

        Args:
            sphere: Sphere image

        Returns:
            Shadow image
        """
        # Create shadow by blurring and darkening the sphere
        shadow = sphere.copy()

        # Convert to grayscale and darken
        shadow_gray = shadow.convert("L")
        shadow_array = np.array(shadow_gray)
        shadow_array = (shadow_array * 0.3).astype(np.uint8)  # Darken

        # Apply blur
        shadow_blurred = Image.fromarray(shadow_array).filter(
            ImageFilter.GaussianBlur(radius=10)
        )

        # Convert back to RGBA with reduced alpha
        shadow_rgba = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
        shadow_rgba.paste(shadow_blurred, mask=shadow_blurred)

        # Reduce alpha
        shadow_array = np.array(shadow_rgba)
        shadow_array[:, :, 3] = (shadow_array[:, :, 3] * 0.5).astype(np.uint8)

        return Image.fromarray(shadow_array)

    def create_multiple_angles(
        self, design_image: Image.Image, angles: int = 4
    ) -> list:
        """
        Create mockups from multiple viewing angles.

        Args:
            design_image: Design image
            angles: Number of angles to generate

        Returns:
            List of mockup images
        """
        mockups = []

        for i in range(angles):
            # Rotate design for different angles
            angle = i * (360 / angles)
            rotated_design = design_image.rotate(angle, expand=True)

            # Create mockup for this angle
            mockup = self.create_mockup(rotated_design)
            mockups.append(mockup)

        return mockups
