# Stable Diffusion ControlNet Pipeline for Custom Print Artwork

A complete AI-powered design engine that generates custom artwork for printed physical products using Stable Diffusion and ControlNet technology.

## 🎯 Project Overview

This pipeline transforms structured prompts into:
- High-quality visual previews with product mockups
- Flattened 2D print files optimized for UV printing workflows

## 📐 Design Constraints

All artwork is confined to a printable band wrapping around a sphere:
- **Band Dimensions:** 134mm wide × 25mm tall
- **Object:** 42.67mm diameter spherical object
- **Requirements:** Clean, centered, undistorted, print-ready for UV RIP software

## 🚀 Features

- **Stable Diffusion Integration:** Support for SD 1.5, 2.1, and XL models
- **ControlNet Implementation:** Layout-constrained generation with image masks
- **Product Mockup Generation:** Realistic 3D object rendering
- **Print File Extraction:** High-resolution flattened designs (300+ DPI)
- **Style Variations:** Photorealistic, illustrative, graphic, minimalist styles
- **Batch Generation:** 2x2 design variant previews

## 🛠️ Technology Stack

- **Python 3.9+**
- **PyTorch** - Deep learning framework
- **Diffusers** - Hugging Face Stable Diffusion library
- **ControlNet** - Layout and structure control
- **Pillow** - Image processing
- **OpenCV** - Computer vision operations
- **NumPy** - Numerical computations

## 📁 Project Structure

```
stable-diffusion-controlnet-pipeline/
├── src/
│   ├── models/           # Model definitions and loading
│   ├── generators/       # Image generation pipelines
│   ├── processors/       # Image processing utilities
│   ├── mockups/          # Product mockup generation
│   └── utils/            # Helper functions
├── configs/              # Configuration files
├── assets/               # Base images and templates
├── outputs/              # Generated images
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── main.py              # Main application entry point
```

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models:**
   ```bash
   python scripts/download_models.py
   ```

3. **Run Generation:**
   ```bash
   python main.py --prompt "A bold design with lightning bolts and flames"
   ```

## 📋 Usage Examples

### Basic Generation
```python
from src.generators.controlnet_pipeline import ControlNetPipeline

pipeline = ControlNetPipeline()
result = pipeline.generate(
    prompt="BEAST MODE in heavy blackletter font with lightning bolts",
    style="graphic",
    output_format="both"  # mockup + flattened
)
```

### Batch Generation
```python
variations = pipeline.generate_variations(
    prompt="Minimalist geometric pattern",
    num_variations=4,
    style="minimalist"
)
```

## 🎨 Supported Styles

- **Photorealistic:** High-detail, realistic rendering
- **Illustrative:** Artistic, hand-drawn aesthetic
- **Graphic:** Bold, geometric, modern designs
- **Minimalist:** Clean, simple, elegant patterns
- **Tattoo-inspired:** Intricate, detailed artwork

## 📊 Output Formats

1. **Product Mockup:** PNG with transparent background
2. **Flattened Design:** 134mm × 25mm @ 300 DPI
3. **Preview Grid:** 2x2 variation layout
4. **Print-Ready:** CMYK optimized for UV printing

## 🔧 Configuration

Edit `configs/pipeline_config.yaml` to customize:
- Model selection and parameters
- Generation settings
- Output formats and quality
- Style presets

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📈 Performance

- **Generation Time:** 15-30 seconds per image
- **Memory Usage:** 8-16GB VRAM recommended
- **Output Quality:** 300+ DPI print-ready
- **Batch Processing:** Up to 4 variations simultaneously

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review example configurations

## 🎯 Roadmap

- [ ] LoRA fine-tuning for brand consistency
- [ ] Web UI interface
- [ ] Real-time preview generation
- [ ] Advanced style transfer
- [ ] Multi-object support 