"""
File utility functions for the pipeline
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import numpy as np


def ensure_output_dir(output_dir: Union[str, Path]) -> Path:
    """
    Ensure output directory exists, create if necessary.

    Args:
        output_dir: Path to output directory

    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def clean_output_dir(output_dir: Union[str, Path], pattern: str = "*") -> None:
    """
    Clean output directory by removing files matching pattern.

    Args:
        output_dir: Path to output directory
        pattern: File pattern to match (default: all files)
    """
    output_path = Path(output_dir)
    if output_path.exists():
        for file_path in output_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)


def save_image(
    image: Union[Image.Image, np.ndarray],
    output_path: Union[str, Path],
    format: str = "PNG",
    quality: int = 95,
    dpi: Optional[tuple] = None,
) -> Path:
    """
    Save image with specified parameters.

    Args:
        image: PIL Image or numpy array
        output_path: Path where to save the image
        format: Image format (PNG, JPEG, etc.)
        quality: Image quality (for JPEG)
        dpi: DPI setting as tuple (x, y)

    Returns:
        Path to saved image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Set DPI if specified
    if dpi:
        image.info["dpi"] = dpi

    # Save image
    if format.upper() == "JPEG":
        image.save(output_path, format=format, quality=quality, optimize=True)
    else:
        image.save(output_path, format=format, optimize=True)

    return output_path


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load image from file path.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path)
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot load image {image_path}: {str(e)}")


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def list_image_files(
    directory: Union[str, Path], extensions: List[str] = None
) -> List[Path]:
    """
    List all image files in directory.

    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: common image formats)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

    directory = Path(directory)
    image_files = []

    if directory.exists() and directory.is_dir():
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def create_preview_grid(
    images: List[Union[Image.Image, np.ndarray]],
    output_path: Union[str, Path],
    grid_size: tuple = (2, 2),
    spacing: int = 10,
    background_color: tuple = (255, 255, 255),
) -> Path:
    """
    Create a grid preview of multiple images.

    Args:
        images: List of images to arrange in grid
        output_path: Path where to save the grid
        grid_size: Grid dimensions (rows, cols)
        spacing: Spacing between images in pixels
        background_color: Background color as RGB tuple

    Returns:
        Path to saved grid image
    """
    if len(images) == 0:
        raise ValueError("No images provided for grid")

    # Convert numpy arrays to PIL Images
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        else:
            pil_images.append(img)

    # Resize all images to same size
    target_size = (512, 512)  # Default size
    resized_images = [
        img.resize(target_size, Image.Resampling.LANCZOS) for img in pil_images
    ]

    # Calculate grid dimensions
    rows, cols = grid_size
    cell_width, cell_height = target_size

    # Create grid image
    grid_width = cols * cell_width + (cols - 1) * spacing
    grid_height = rows * cell_height + (rows - 1) * spacing

    grid_image = Image.new("RGB", (grid_width, grid_height), background_color)

    # Place images in grid
    for i, img in enumerate(resized_images[: rows * cols]):
        row = i // cols
        col = i % cols

        x = col * (cell_width + spacing)
        y = row * (cell_height + spacing)

        grid_image.paste(img, (x, y))

    # Save grid
    return save_image(grid_image, output_path)


def validate_image_dimensions(
    image: Union[Image.Image, np.ndarray],
    expected_width: int,
    expected_height: int,
    tolerance: int = 10,
) -> bool:
    """
    Validate image dimensions against expected values.

    Args:
        image: Image to validate
        expected_width: Expected width in pixels
        expected_height: Expected height in pixels
        tolerance: Acceptable deviation in pixels

    Returns:
        True if dimensions are within tolerance
    """
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        width, height = image.size

    width_ok = abs(width - expected_width) <= tolerance
    height_ok = abs(height - expected_height) <= tolerance

    return width_ok and height_ok
