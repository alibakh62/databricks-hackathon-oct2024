# AI Image Generation Expert (Stable Diffusion + ControlNet) for Custom Print Artwork

## Job Description

We’re building an AI-powered design engine that allows users to create custom artwork for a printed physical product using a bot assistant. These designs vary in style — from photorealistic to illustrative, tattoo-inspired, graphic, or minimalist — depending on the user’s creative direction.

We are looking for an expert in **Stable Diffusion + ControlNet** to help us build a generation pipeline that transforms structured prompts into both:
- a high-quality visual preview, and
- a flattened 2D print file that fits very specific physical constraints.

> **Important Note:**  
You will not be responsible for chatbot development.  
You will receive structured prompts directly from another system — your job is to take those prompts and generate accurate artwork and mockups.

---

## What You’ll Build

1. A complete Stable Diffusion + ControlNet image generation pipeline (local or hosted)
2. Image outputs that include:
   - A product mockup using our supplied 3D object base image (spherical)
   - A flattened 2D design file suitable for UV printing

---

## Design Constraints

All artwork must be strictly confined to a printable band that wraps around a sphere:

**Band Dimensions:**  
`134mm wide × 25mm tall`  
(Wrapping around a 42.67mm diameter spherical object)

**Artwork must:**
- Stay within the band boundaries  
- Be clean, centered, and undistorted  
- Be print-ready for a UV RIP software workflow (we’ll define specs)

---

## Example Workflow

1. You receive a prompt like:

   > “A bold design with lightning bolts, flames, and the word ‘BEAST MODE’ in heavy blackletter font.”

2. Your system:
   - Generates a photorealistic or stylized mockup of that design applied to the equator band of our product
   - Extracts the flattened design band (134mm × 25mm) as a separate high-resolution image (PNG or similar)

3. (Optional) You generate 4 variations in a 2x2 preview grid, numbered for user selection.

---

## Requirements

**Proven experience with:**
- Stable Diffusion 1.5 / 2.1 / XL  
- ControlNet with image masks or reference guides  

**Familiarity with:**
- Layout-constrained generation workflows  
- Product rendering and surface-aware design  
- Styles: photorealism, illustration, iconography, type-heavy designs  

**Ability to:**
- Extract flattened 2D print files from curved mockup outputs  
- Ensure spelling accuracy and print-safe alignment  

**Bonus:**
- Experience with AUTOMATIC1111 or ComfyUI  
- LoRA fine-tuning for consistent brand style output  
- Prior work with print-on-demand or UV direct-to-object workflows  

---

## Deliverables

- A fully functioning generation pipeline that accepts text prompts  
- Consistent design previews (mockups)  
- Flattened band output for printing (134mm × 25mm @ 300+ DPI)  
- Documentation and reusable templates  
- (Optional) 2x2 design variant generation  

---

## Budget

💰 **$2,000 – $4,000**, depending on experience and scope  
Open to hourly or fixed-price structure with defined milestones.

---

## Timeline

We’re ready to start **immediately**.  
We’re targeting a **prototype-ready solution in 2–4 weeks.**

---

## To Apply

Please include:
- Samples of previous work with Stable Diffusion and layout constraints  
- Any experience with mockup-to-print pipelines  
- Your approach to building and automating this type of workflow
