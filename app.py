import gradio as gr
import tempfile
from campaign_gen import run_research_agent
import markdown
import os
import traceback
import yaml
from datetime import datetime
import requests
from image_gen import generate_image
import imageio
import numpy as np
import replicate
from dotenv import load_dotenv
import json
from google.cloud import storage
from utils import GCSBucketManager, ModelConfig
from image_gen import (
    create_replicate_model,
    start_training_job,
    monitor_training_jobs,
    get_model_details
)
import time

# Load environment variables from .env file
load_dotenv()

# Initialize the Replicate client
client = replicate.Client(api_token=os.getenv("REPLICATE_API_KEY"))


def load_models():
    with open("models.yaml", "r") as f:
        models = yaml.safe_load(f)
        # Get the first model as default
        default_model = list(models["models"].keys())[0]
        return models["models"], default_model


def generate_report(product_desc, campaign_desc, industry, progress=gr.Progress()):
    try:
        progress(0, desc="Initializing research agent...")
        progress(0.2, desc="Generating research report...")
        campaign_info, generated_email = run_research_agent(
            product_desc, campaign_desc, industry
        )
        progress(1.0, desc="Done!")
        return (
            campaign_info,
            gr.update(visible=True),  # Download Report button
            gr.update(visible=True),  # Generate Email button
            gr.update(visible=False),  # Copy Email button
            generated_email,  # Store the email but don't show it yet
            gr.update(visible=False),  # processing_status
        )
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return (
            f"An error occurred while generating the report. Please try again or contact support if the issue persists.",
            gr.update(visible=False),  # Download Report button
            gr.update(visible=False),  # Generate Email button
            gr.update(visible=False),  # Copy Email button
            "",  # email_output
            gr.update(visible=False),  # processing_status
        )


def markdown_to_pdf(text, progress=gr.Progress()):
    # ... rest of the markdown_to_pdf function ...
    pass


def show_email(email_text):
    if email_text:
        return gr.update(value=email_text, visible=True), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)


def clear_inputs():
    return {
        product_input: "",
        campaign_input: "",
        industry_input: "",
        report_output: "",
        email_output: gr.update(value="", visible=False),
        download_button: gr.update(visible=False),
        email_button: gr.update(visible=False),
        copy_button: gr.update(visible=False),
        processing_status: gr.update(visible=False),
        pdf_output: gr.update(value=None, visible=False),
    }


def save_generated_image(image_url):
    """Save image from URL to local file"""
    os.makedirs("generated_images", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png"
    filepath = os.path.join("generated_images", filename)

    response = requests.get(image_url)
    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


def generate_images(prompt, model_name, num_outputs):
    # Create generated_images directory if it doesn't exist
    os.makedirs("generated_images", exist_ok=True)

    # Get the actual model identifier from the models dictionary
    models, _ = load_models()
    model_name = models[model_name]

    # Generate images using the full model identifier
    output = generate_image(prompt, model_name, int(num_outputs))

    # Save images and collect absolute paths
    image_paths = []
    for i, item in enumerate(output):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(f"generated_images/generated_image_{i}_{timestamp}.png")
        with open(path, "wb") as f:
            f.write(item.read())
        image_paths.append(path)
    print(f"Generated images: {image_paths}")
    return image_paths


def clear_image_inputs():
    return {
        image_prompt: "",
        image_output: None,
        action_buttons: gr.update(visible=False),
        processing_status: gr.update(visible=False),
    }


class GCSBucketManager:
    def __init__(self, bucket_name, credentials_path):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.storage_client = storage.Client.from_service_account_json(self.credentials_path)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def upload_file(self, source_path, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_path)
        return destination_blob_name

    def make_blob_public(self, blob_name):
        blob = self.bucket.blob(blob_name)
        blob.make_public()
        return blob.public_url

    def get_public_url(self, blob_name):
        blob = self.bucket.blob(blob_name)
        return blob.public_url


with gr.Blocks() as app:
    with gr.Tab("Generate Report"):
        with gr.Row():
            with gr.Column():
                product_input = gr.Textbox(
                    label="Product Description",
                    placeholder="Enter product description...",
                )
                campaign_input = gr.Textbox(
                    label="Campaign Description",
                    placeholder="Enter campaign description...",
                )
                industry_input = gr.Textbox(
                    label="Industry", placeholder="Enter industry..."
                )
                generate_button = gr.Button("Generate Report")
                clear_button = gr.Button("Clear")

        with gr.Row():
            processing_status = gr.Markdown("Processing... Please wait.", visible=False)

        with gr.Row():
            report_output = gr.Markdown()

        with gr.Row():
            download_button = gr.Button("Download Report", visible=False)
            email_button = gr.Button("Generate Email", visible=False)

        pdf_output = gr.File(
            label="Download PDF", visible=False, interactive=True, type="filepath"
        )

        with gr.Row():
            email_output = gr.Markdown(visible=False)

        with gr.Row():
            copy_button = gr.Button("Copy Email", visible=False)

    with gr.Tab("Generate Image"):
        # Load models from yaml
        models, default_model = load_models()

        with gr.Column():
            # Prompt input
            prompt_input = gr.Textbox(
                label="Image Prompt", placeholder="Enter your prompt here..."
            )

            # Settings section
            with gr.Accordion("Settings", open=False):
                model_dropdown = gr.Dropdown(
                    choices=list(models.keys()), value=default_model, label="Model"
                )
                num_outputs = gr.Dropdown(
                    choices=["1"],  # Since we're only showing one image
                    value="1",
                    label="Number of Outputs",
                )

            # Generate button
            generate_btn = gr.Button("Generate Image")

            # Single image output
            image_output = gr.Image(
                label="Generated Image", show_label=True, type="filepath", visible=True
            )

            # Fine-tune and Edit buttons (hidden by default)
            with gr.Row(visible=False) as action_buttons:
                fine_tune_btn = gr.Button("Fine-tune")
                edit_btn = gr.Button("Edit Image")

            # Clear button
            clear_btn = gr.Button("Clear")

            # Handle generate button click
            def on_generate(prompt, model, num_out):
                try:
                    image_paths = generate_images(prompt, model, int(num_out))
                    print(f"Generated images (on_generate): {image_paths}")
                    # Return just the first image path since we're only showing one image
                    return {
                        image_output: image_paths[0],
                        action_buttons: gr.update(visible=True),
                    }
                except Exception as e:
                    print(f"Error generating images: {str(e)}")
                    return {
                        image_output: None,
                        action_buttons: gr.update(visible=False),
                    }

            generate_btn.click(
                fn=on_generate,
                inputs=[prompt_input, model_dropdown, num_outputs],
                outputs=[image_output, action_buttons],
            )

            # Handle clear button click
            def clear_all():
                return {
                    prompt_input: "",
                    image_output: None,
                    action_buttons: gr.update(visible=False),
                }

            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[prompt_input, image_output, action_buttons],
            )

            # Handle fine-tune and edit buttons
            def switch_to_tab(tab_index):
                # This will be implemented when we add the other tabs
                pass

            fine_tune_btn.click(lambda: switch_to_tab(2))
            edit_btn.click(lambda: switch_to_tab(1))

    with gr.Tab("Edit Image"):
        with gr.Column():
            # Original image upload/editor with mask
            img_editor = gr.ImageMask(
                label="Edit Image",
                sources=["upload"],
                height=1024,
                width=1024,
                layers=False,
                transforms=[],
                format="png",
                show_label=True,
            )

            # Mask prompt input
            mask_prompt = gr.Textbox(
                label="Edit Prompt",
                placeholder="Describe what you want to change in the selected area...",
            )

            # Apply edit button
            apply_edit_btn = gr.Button("Apply Edit", interactive=False)

            # Processing indicator
            processing_indicator = gr.Markdown(
                "Processing... Please wait.", visible=False
            )

            # Results gallery for inpainted images
            inpainted_gallery = gr.Gallery(
                label="Generated Variations",
                show_label=True,
                columns=2,
                rows=2,
                height="auto",
                visible=True,  # Make it always visible
                allow_preview=True,  # Enable image preview on click
                preview=True,  # Show preview mode
                object_fit="contain"  # Ensure images fit properly
            )

            # Download button for results
            download_btn = gr.Button("Download Selected Images", visible=False)

            def enable_edit_button(img):
                print("Image uploaded/changed. Enabling edit button...")
                return gr.update(interactive=True if img is not None else False)

            def save_and_process_image(img):
                print("Starting save_and_process_image...")
                if img is None:
                    print("No image provided")
                    return None, None

                try:
                    print("Saving composite image and generating mask...")
                    # Save the composite image
                    composite_image = img["composite"]
                    imageio.imwrite("input_image.png", composite_image)

                    # Extract and save the mask
                    alpha_channel = img["layers"][0][:, :, 3]
                    mask = np.where(alpha_channel == 0, 0, 255).astype(np.uint8)
                    imageio.imwrite("mask_image.png", mask)

                    print("Image and mask saved successfully")
                    return "input_image.png", "mask_image.png"
                except Exception as e:
                    print(f"Error in save_and_process_image: {str(e)}")
                    return None, None

            def apply_edit(img, prompt):
                try:
                    print("Starting apply_edit function...")
                    
                    # First save and process the image/mask
                    input_path, mask_path = save_and_process_image(img)
                    if not input_path or not mask_path:
                        print("Failed to save image or mask")
                        return (
                            [],  # Empty list for gallery instead of None
                            gr.update(visible=False),  # inpainted_gallery visibility
                            gr.update(visible=False),  # download_btn visibility
                            gr.update(value="", visible=False),  # processing_indicator
                        )

                    print("Starting inpainting process...")
                    # Show processing results
                    with open(input_path, "rb") as f_img, open(mask_path, "rb") as f_mask:
                        input = {
                            "mask": f_mask,
                            "image": f_img,
                            "prompt": prompt,
                            "num_outputs": 4,
                            "output_format": "png",
                        }

                        print("Calling Replicate API...")
                        # Call the inpainting API
                        output = client.run(
                            "zsxkib/flux-dev-inpainting-controlnet:f9cb02cfd6b131af7ff9166b4bac5fdd2ed68bc282d2c049b95a23cea485e40d",
                            input=input,
                        )

                        print("Saving inpainted results...")
                        # Save inpainted results with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        inpainted_paths = []
                        
                        # Create directory if it doesn't exist
                        os.makedirs("inpainted_images", exist_ok=True)
                        
                        for i, item in enumerate(output):
                            path = os.path.abspath(f"inpainted_images/inpainted_image_{i}_{timestamp}.png")
                            with open(path, "wb") as f:
                                f.write(item.read())
                            inpainted_paths.append(path)
                            print(f"Saved inpainted image {i} at: {path}")

                        print("All inpainted images saved. Paths:", inpainted_paths)
                        
                        # Return the paths for the gallery along with other updates
                        return (
                            inpainted_paths,  # List of paths for gallery
                            gr.update(visible=True),  # inpainted_gallery visibility
                            gr.update(visible=True),  # download_btn visibility
                            gr.update(value="Processing complete!", visible=False)  # processing indicator
                        )

                except Exception as e:
                    print(f"Error in apply_edit: {str(e)}")
                    traceback.print_exc()  # Print full traceback
                    return (
                        [],  # Empty list for gallery instead of None
                        gr.update(visible=False),  # inpainted_gallery visibility
                        gr.update(visible=False),  # download_btn visibility
                        gr.update(value=f"Error: {str(e)}", visible=True)  # processing indicator
                    )

            # Event handlers
            img_editor.change(
                fn=enable_edit_button, 
                inputs=[img_editor], 
                outputs=[apply_edit_btn]
            )

            # Add a loading status to the apply_edit button click
            apply_edit_btn.click(
                fn=lambda: ([], gr.update(visible=True), gr.update(visible=False), gr.update(value="Processing... Please wait.", visible=True)),
                inputs=None,
                outputs=[inpainted_gallery, inpainted_gallery, download_btn, processing_indicator],
                queue=False
            ).then(
                fn=apply_edit,
                inputs=[img_editor, mask_prompt],
                outputs=[
                    inpainted_gallery,
                    inpainted_gallery,
                    download_btn,
                    processing_indicator,
                ]
            )

    with gr.Tab("Fine-tune Model"):
        with gr.Column():
            # Note text at the top
            gr.Markdown("""
                The fine-tuning process is done by uploading a file to the model. The file should be a zip file that contains 
                the images to fine-tune the model. The name of files or their aspect ratio don't matter. The zip file should 
                contain at least 10 images. The images should be in the PNG format. The proper resolution for the images is 1024x1024.
            """)

            # Settings section in a collapsible area
            with gr.Accordion("Settings", open=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name_input = gr.Textbox(
                    label="Model Name",
                    value=f"finetuned_inpainting_{timestamp}",
                    placeholder="Enter model name..."
                )
                trigger_word_input = gr.Textbox(
                    label="Trigger Word",
                    value="MARKETER",
                    placeholder="Enter trigger word..."
                )
                description_input = gr.Textbox(
                    label="Description",
                    value="a fine-tuned image generation model",
                    placeholder="Enter model description..."
                )

            # File upload
            file_upload = gr.File(
                label="Upload Training Images (ZIP)",
                file_types=[".zip"],
                type="filepath"
            )

            # Start fine-tuning button (disabled by default)
            start_finetune_btn = gr.Button("Start Fine-Tuning", interactive=False)

            # Status display
            status_output = gr.Markdown(visible=False)

            def enable_start_button(file):
                return gr.update(interactive=True if file is not None else False)

            def start_finetuning(file_path, model_name, trigger_word, description):
                try:
                    print("Starting fine-tuning process...")
                    
                    # Upload file to GCS
                    print("Uploading file to GCS...")
                    bucket_name = os.getenv("GCS_BUCKET_NAME")
                    credentials_path = os.getenv("GCS_CREDENTIALS_PATH")

                    bucket_manager = GCSBucketManager(
                        bucket_name=bucket_name,
                        credentials_path=credentials_path
                    )

                    # Get the filename from the path
                    filename = os.path.basename(file_path)
                    
                    # Upload the file
                    uploaded_blob_name = bucket_manager.upload_file(
                        source_path=file_path,
                        destination_blob_name=filename
                    )
                    print(f"File uploaded as: {uploaded_blob_name}")

                    # Make it public and get URL
                    public_url = bucket_manager.make_blob_public(uploaded_blob_name)
                    url = bucket_manager.get_public_url(uploaded_blob_name)
                    print(f"Public URL: {url}")

                    # Step 3.4.1 - Create the destination model
                    print("\nStep 3.4.1 - Creating destination model...")
                    try:
                        username = os.getenv("USERNAME")
                        if not username:
                            raise ValueError("USERNAME environment variable not set")
                        
                        result = create_replicate_model(
                            username=username,
                            model_name=model_name,
                            description=description
                        )
                        print("Model created successfully:", json.dumps(result, indent=2))
                        
                    except Exception as e:
                        print(f"Failed to create model: {str(e)}")
                        return gr.update(
                            value=f"❌ Failed to create model: {str(e)}",
                            visible=True
                        )

                    # Step 3.4.2 - Start the training job
                    print("\nStep 3.4.2 - Starting training job...")
                    try:
                        result = start_training_job(
                            username=username,
                            model_name=model_name,
                            data_url=url,
                            trigger_word=trigger_word
                        )
                        print("Training job response:", json.dumps(result, indent=2))
                        
                        if not result:
                            raise ValueError("Training job returned no result")
                        
                        # Step 3.4.3 - Monitor the training process
                        print("\nStep 3.4.3 - Monitoring training process...")
                        print("Waiting 10 seconds for training job to initialize...")
                        time.sleep(10)  # Give the job time to start
                        
                        final_statuses = monitor_training_jobs(check_interval=60)
                        print("Monitoring returned with statuses:", json.dumps(final_statuses, indent=2))
                        
                        if not final_statuses:
                            print("Warning: No job statuses returned from monitoring")
                            # Continue anyway as the job might still be successful
                        
                        print("\nFinal Status Summary:")
                        print("-" * 40)
                        for job_id, status in final_statuses.items():
                            print(f"Job {job_id}: {status}")
                        
                        # Add a delay before getting model details
                        print("Waiting 30 seconds for model to finalize...")
                        time.sleep(30)
                        
                    except Exception as e:
                        print(f"Failed to start/monitor training: {str(e)}")
                        print(f"Full error details: {traceback.format_exc()}")
                        return gr.update(
                            value=f"❌ Failed during training process: {str(e)}",
                            visible=True
                        )

                    # Step 3.4.1 - Get model details
                    print("\nGetting model details...")
                    try:
                        model_details = get_model_details(
                            owner=username,
                            model_name=model_name
                        )
                        
                        if not model_details:
                            raise ValueError("Failed to get model details")
                            
                        model_url = model_details.get("url")
                        if not model_url:
                            raise ValueError("Model URL not found in response")
                            
                        model_id = model_details.get("latest_version", {}).get("id")
                        if not model_id:
                            raise ValueError("Model ID not found in response")

                        # Step 3.4.2 - Update models.yaml
                        print("\nUpdating models.yaml...")
                        model_config = ModelConfig('models.yaml')
                        model_config.add_or_update_model(
                            model_name.replace("-", "_"),
                            f"{model_url}?:{model_id}"
                        )

                        return gr.update(
                            value="✅ Fine-tuning process completed successfully! Please go back to the image generation UI (Tab 1) and generate an image with the new model.",
                            visible=True
                        )

                    except Exception as e:
                        print(f"Failed to update model configuration: {str(e)}")
                        return gr.update(
                            value=f"❌ Failed to update model configuration: {str(e)}",
                            visible=True
                        )

                except Exception as e:
                    print(f"Error in fine-tuning process: {str(e)}")
                    traceback.print_exc()
                    return gr.update(
                        value=f"❌ Error during fine-tuning: {str(e)}",
                        visible=True
                    )

            # Event handlers
            file_upload.change(
                fn=enable_start_button,
                inputs=[file_upload],
                outputs=[start_finetune_btn]
            )

            start_finetune_btn.click(
                fn=lambda: gr.update(value="⏳ Fine-tuning process started. This may take several minutes...", visible=True),
                inputs=None,
                outputs=status_output,
                queue=False
            ).then(
                fn=start_finetuning,
                inputs=[
                    file_upload,
                    model_name_input,
                    trigger_word_input,
                    description_input
                ],
                outputs=[status_output]
            )

if __name__ == "__main__":
    app.launch()
