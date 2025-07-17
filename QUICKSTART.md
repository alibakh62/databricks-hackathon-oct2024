# Quick Start Guide

Get up and running with the Stable Diffusion ControlNet Pipeline in minutes!

## 🚀 Prerequisites

- **Python 3.9+** installed
- **8GB+ RAM** (16GB+ recommended)
- **GPU with 8GB+ VRAM** (for optimal performance)
- **Internet connection** (for model downloads)

## 📦 Installation

### Option 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/stable-diffusion-controlnet-pipeline.git
cd stable-diffusion-controlnet-pipeline

# Install dependencies
pip install -r requirements.txt

# Download models (this may take 10-20 minutes)
python scripts/download_models.py
```

### Option 2: Development Install

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/stable-diffusion-controlnet-pipeline.git
cd stable-diffusion-controlnet-pipeline

# Install with development dependencies
pip install -e .[dev,gpu]
```

## 🎯 First Run

### Basic Generation

```bash
# Generate a simple design
python main.py --prompt "BEAST MODE in bold letters" --style graphic

# Generate with variations
python main.py --prompt "Minimalist geometric pattern" --style minimalist --variations 4
```

### Advanced Usage

```bash
# Generate photorealistic design
python main.py \
  --prompt "Mountain landscape at sunset" \
  --style photorealistic \
  --output-format both \
  --seed 42

# Generate tattoo-style design
python main.py \
  --prompt "Traditional skull and roses tattoo" \
  --style tattoo \
  --variations 4
```

## 📁 Output Files

After generation, you'll find:

- **Mockup**: `outputs/mockup_graphic_42.png` - 3D product visualization
- **Flattened**: `outputs/flattened_graphic_42.png` - Print-ready band design
- **Preview Grid**: `outputs/preview_grid.png` - 2x2 variation overview

## ⚙️ Configuration

Edit `configs/pipeline_config.yaml` to customize:

```yaml
generation:
  num_inference_steps: 20    # More steps = better quality, slower
  guidance_scale: 7.5        # Higher = more prompt adherence
  width: 512                 # Image width
  height: 512                # Image height

output:
  dpi: 300                   # Print resolution
  band_width_mm: 134         # Band width in mm
  band_height_mm: 25         # Band height in mm
```

## 🎨 Available Styles

- **`graphic`** - Bold, modern graphic designs
- **`minimalist`** - Clean, simple patterns
- **`photorealistic`** - High-detail realistic images
- **`illustrative`** - Artistic, hand-drawn style
- **`tattoo`** - Traditional tattoo artwork

## 🔧 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Reduce image size or batch size
python main.py --prompt "test" --style graphic
# Edit config: width: 384, height: 384
```

**"Model not found"**
```bash
# Re-download models
python scripts/download_models.py --force
```

**"Slow generation"**
```bash
# Enable memory optimizations in config
system:
  enable_xformers: true
  enable_attention_slicing: true
```

### Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed
2. **Optimize memory**: Enable xformers and attention slicing
3. **Reduce quality**: Lower inference steps for faster generation
4. **Batch processing**: Generate multiple variations at once

## 📚 Next Steps

- Read the [full documentation](README.md)
- Check [examples](examples/) for advanced usage
- Run [tests](tests/) to verify installation
- Customize [styles](configs/pipeline_config.yaml#styles) for your needs

## 🆘 Need Help?

- Check the [troubleshooting section](#troubleshooting)
- Review [example configurations](configs/)
- Open an [issue](https://github.com/yourusername/stable-diffusion-controlnet-pipeline/issues)

---

**Happy generating! 🎨✨** 