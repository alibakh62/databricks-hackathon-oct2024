#!/usr/bin/env python3
"""
Setup script for Stable Diffusion ControlNet Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="stable-diffusion-controlnet-pipeline",
    version="1.0.0",
    author="AI Image Generation Expert",
    author_email="",
    description="AI-powered design engine for custom print artwork using Stable Diffusion and ControlNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stable-diffusion-controlnet-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "xformers>=0.0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "sd-controlnet-pipeline=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "stable-diffusion",
        "controlnet",
        "ai-art",
        "image-generation",
        "print-design",
        "product-mockup",
        "machine-learning",
        "deep-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stable-diffusion-controlnet-pipeline/issues",
        "Source": "https://github.com/yourusername/stable-diffusion-controlnet-pipeline",
        "Documentation": "https://github.com/yourusername/stable-diffusion-controlnet-pipeline#readme",
    },
)
