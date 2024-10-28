# Create the destination model
import os
import replicate
from dotenv import load_dotenv

load_dotenv()


### Generate Image
def generate_image(prompt: str, model: str, num_outputs: int = 1):
    """
    Generate images using Replicate API
    Args:
        prompt (str): Text prompt for image generation
        model (str): Model ID to use
        num_outputs (int): Number of images to generate
    Returns:
        list: List of generated image URLs
    """
    client = replicate.Client(api_token=os.getenv("REPLICATE_API_KEY"))
    if model.startswith("black-forest-labs"):
        prompt = f'{prompt} "FLUX SCHNELL"'

    output = client.run(
        model,
        input={
            "prompt": prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": num_outputs,
            "aspect_ratio": "1:1",
            "output_format": "png",
            "output_quality": 80,
            "num_inference_steps": 4,
        },
    )
    return output
