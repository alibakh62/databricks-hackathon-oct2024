import gradio as gr
import tempfile
from campaign_gen import run_research_agent
import markdown
import os
import traceback
import yaml
from datetime import datetime
import requests

def load_models():
    try:
        with open('models.yaml', 'r') as file:
            models = yaml.safe_load(file)
        return models.get('models', {})
    except:
        return {}

def generate_report(product_desc, campaign_desc, industry, progress=gr.Progress()):
    try:
        progress(0, desc="Initializing research agent...")
        progress(0.2, desc="Generating research report...")
        campaign_info, generated_email = run_research_agent(product_desc, campaign_desc, industry)
        progress(1.0, desc="Done!")
        return (
            campaign_info, 
            gr.update(visible=True),  # Download Report button
            gr.update(visible=True),  # Generate Email button
            gr.update(visible=False), # Copy Email button
            generated_email,  # Store the email but don't show it yet
            gr.update(visible=False)  # processing_status
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
            gr.update(visible=False)  # processing_status
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
        pdf_output: gr.update(value=None, visible=False)
    }

def save_generated_image(image_url):
    """Save image from URL to local file"""
    os.makedirs("generated_images", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png"
    filepath = os.path.join("generated_images", filename)
    
    response = requests.get(image_url)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    return filepath

def generate_images(prompt, model, num_outputs):
    try:
        from image_gen import generate_image
        
        # Show processing status
        yield gr.update(visible=True), None, gr.update(visible=False)
        
        # Generate images
        image_urls = generate_image(prompt, model, int(num_outputs))
        
        # Save images locally
        saved_paths = []
        for url in image_urls:
            saved_path = save_generated_image(url)
            saved_paths.append(saved_path)
        
        # Hide processing status and show images
        return gr.update(visible=False), saved_paths, gr.update(visible=True)
        
    except Exception as e:
        error_msg = f"Error generating images: {str(e)}"
        print(error_msg)
        return gr.update(value=error_msg, visible=True), None, gr.update(visible=False)

def clear_image_inputs():
    return {
        image_prompt: "",
        image_output: None,
        action_buttons: gr.update(visible=False),
        processing_status: gr.update(visible=False)
    }

with gr.Blocks() as app:
    with gr.Tab("Generate Report"):
        with gr.Row():
            with gr.Column():
                product_input = gr.Textbox(
                    label="Product Description", 
                    placeholder="Enter product description..."
                )
                campaign_input = gr.Textbox(
                    label="Campaign Description", 
                    placeholder="Enter campaign description..."
                )
                industry_input = gr.Textbox(
                    label="Industry", 
                    placeholder="Enter industry..."
                )
                generate_button = gr.Button("Generate Report")
                clear_button = gr.Button("Clear")
                
        with gr.Row():
            processing_status = gr.Markdown(
                "Processing... Please wait.", 
                visible=False
            )
            
        with gr.Row():
            report_output = gr.Markdown()
            
        with gr.Row():
            download_button = gr.Button("Download Report", visible=False)
            email_button = gr.Button("Generate Email", visible=False)
            
        pdf_output = gr.File(
            label="Download PDF", 
            visible=False, 
            interactive=True,
            type="filepath"
        )
            
        with gr.Row():
            email_output = gr.Markdown(visible=False)
            
        with gr.Row():
            copy_button = gr.Button("Copy Email", visible=False)

    with gr.Tab("Generate Image"):
        with gr.Column():
            image_prompt = gr.Textbox(
                label="Image Prompt",
                placeholder="Enter your image prompt..."
            )
            
            with gr.Accordion("Settings", open=False):
                models = load_models()
                model_names = list(models.keys())
                model_dropdown = gr.Dropdown(
                    choices=model_names,
                    value=model_names[0] if model_names else None,
                    label="Model"
                )
                num_outputs = gr.Dropdown(
                    choices=[str(i) for i in range(1, 11)],
                    value="1",
                    label="Number of Outputs"
                )
            
            processing_status_img = gr.Markdown(
                "Generating image... Please wait.",
                visible=False
            )
            
            generate_img_button = gr.Button("Generate Image")
            
            image_output = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                height="auto"
            )
            
            with gr.Row(visible=False) as action_buttons:
                fine_tune_btn = gr.Button("Fine-tune")
                edit_image_btn = gr.Button("Edit Image")
            
            clear_img_button = gr.Button("Clear")

        # Event handlers for report generation
        generate_button.click(
            fn=lambda: gr.update(visible=True),
            outputs=[processing_status],
        ).then(
            generate_report,
            inputs=[product_input, campaign_input, industry_input],
            outputs=[
                report_output, 
                download_button,
                email_button,
                copy_button,
                email_output,
                processing_status
            ]
        )
        
        clear_button.click(
            clear_inputs,
            outputs=[
                product_input, 
                campaign_input, 
                industry_input, 
                report_output, 
                email_output,
                download_button,
                email_button,
                copy_button,
                processing_status,
                pdf_output
            ]
        )
        
        download_button.click(
            markdown_to_pdf,
            inputs=[report_output],
            outputs=[pdf_output]
        )
        
        email_button.click(
            show_email,
            inputs=[email_output],
            outputs=[email_output, copy_button]
        )
        
        # Event handlers for image generation
        generate_img_button.click(
            generate_images,
            inputs=[image_prompt, model_dropdown, num_outputs],
            outputs=[processing_status_img, image_output, action_buttons]
        )
        
        clear_img_button.click(
            clear_image_inputs,
            outputs=[image_prompt, image_output, action_buttons, processing_status_img]
        )

if __name__ == "__main__":
    app.launch()
