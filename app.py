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

if __name__ == "__main__":
    app.launch()
