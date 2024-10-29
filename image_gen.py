# Create the destination model
import os
import requests
import json
import replicate
from dotenv import load_dotenv
import time
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any

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


def create_replicate_model(
    username: str,
    model_name: str,
    description: str = "An example model",
    visibility: str = "public",
    hardware: str = "gpu-a40-large",
) -> dict:
    """
    Create a new model on Replicate using the API.
    """
    try:
        # Get API token from environment variables
        api_token = os.getenv("REPLICATE_API_KEY")
        if not api_token:
            raise KeyError("REPLICATE_API_KEY environment variable not found")

        print(f"Creating model with username={username}, model_name={model_name}")

        # API endpoint
        url = "https://api.replicate.com/v1/models"

        # Request headers
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        # Request payload
        payload = {
            "owner": username,
            "name": model_name,
            "description": description,
            "visibility": visibility,
            "hardware": hardware,
        }

        print("Making API request with payload:", json.dumps(payload, indent=2))

        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)

        # Raise an exception for bad status codes
        response.raise_for_status()

        result = response.json()
        print("Model creation response:", json.dumps(result, indent=2))
        return result

    except requests.exceptions.RequestException as e:
        print(f"API Error during model creation: {str(e)}")
        if hasattr(e.response, "text"):
            print(f"Response text: {e.response.text}")
        raise
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def list_training_jobs(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List training jobs from Replicate API with optional status filtering.
    
    Args:
        status_filter (str, optional): Filter jobs by status (e.g., "processing", "succeeded", "failed")
    
    Returns:
        List[Dict[str, Any]]: List of training jobs
    
    Raises:
        requests.exceptions.RequestException: If the API request fails
        KeyError: If REPLICATE_API_TOKEN environment variable is not set
    """
    
    # Get API token
    api_token = os.getenv('REPLICATE_API_KEY')
    if not api_token:
        raise KeyError("REPLICATE_API_KEY environment variable not found")
    
    # API endpoint
    url = "https://api.replicate.com/v1/trainings"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    
    try:
        # Make the GET request
        response = requests.get(
            url,
            headers=headers
        )
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        data = response.json()
        
        # Filter results if status_filter is provided
        if status_filter:
            data['results'] = [
                training for training in data.get('results', [])
                if training.get('status') == status_filter
            ]
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error listing training jobs: {str(e)}")
        raise

def start_training_job(
    username: str,
    model_name: str,
    data_url: str,
    trigger_word: str,
    version_id: str = "d995297071a44dcb72244e6c19462111649ec86a9646c32df56daa7f14801944",
) -> dict:
    """Start a fine-tuning job on Replicate using direct API calls."""
    try:
        # Get API token from environment variables
        api_token = os.getenv("REPLICATE_API_KEY")
        if not api_token:
            raise KeyError("REPLICATE_API_KEY environment variable not found")

        print(f"Starting training with username={username}, model_name={model_name}")

        # API endpoint for trainings
        url = f"https://api.replicate.com/v1/models/ostris/flux-dev-lora-trainer/versions/{version_id}/trainings"

        # Request headers
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        # Request payload
        payload = {
            "destination": f"{username}/{model_name}",
            "input": {"input_images": data_url, "trigger_word": trigger_word},
        }

        print(
            "Making training API request with payload:", json.dumps(payload, indent=2)
        )

        # Make the POST request
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error starting training: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

def format_training_details(trainings: Dict[str, Any]) -> None:
    """
    Print formatted training job details.
    
    Args:
        trainings (Dict[str, Any]): API response containing training jobs
    """
    results = trainings.get('results', [])
    
    if not results:
        print("\nNo training jobs found matching the criteria.")
        return
        
    print(f"\nFound {len(results)} training job(s):")
    print("-" * 80)
    
    for training in results:
        # Extract basic information
        status = training.get('status', 'Unknown')
        created_at = training.get('created_at', '')
        
        # Convert UTC timestamp to datetime if it exists
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = created_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            except ValueError:
                pass
        
        print(f"ID: {training.get('id', 'N/A')}")
        print(f"Status: {status}")
        print(f"Created: {created_at}")
        print(f"Model: {training.get('model', 'N/A')}")
        print(f"Version: {training.get('version', 'N/A')}")
        
        # Print input parameters if they exist
        if 'input' in training:
            print("Input Parameters:")
            for key, value in training['input'].items():
                print(f"  {key}: {value}")
        
        # Print URLs if they exist
        if 'urls' in training:
            print("URLs:")
            for key, value in training['urls'].items():
                print(f"  {key}: {value}")
        
        print("-" * 80)

def get_processing_jobs(status_filter=None):
    """
    Convenience function to get only processing jobs.
    """
    if status_filter is None:
        return list_training_jobs()
    else:
        return list_training_jobs(status_filter=status_filter)

def get_training_status(training_id: str) -> str:
    """
    Get the status of a specific training job.
    
    Args:
        training_id (str): The ID of the training job
        
    Returns:
        str: Current status of the training job
    """
    api_token = os.getenv('REPLICATE_API_KEY')
    if not api_token:
        raise KeyError("REPLICATE_API_KEY environment variable not found")
    
    url = f"https://api.replicate.com/v1/trainings/{training_id}"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get('status')


def monitor_training_jobs(check_interval=60):
    """Monitor active training jobs until completion."""
    try:
        print("\nStarting monitor_training_jobs function...")
        
        # Get processing jobs
        print("Getting list of processing jobs...")
        trainings = get_processing_jobs(status_filter="processing")
        print(f"Raw trainings response: {json.dumps(trainings, indent=2)}")
        
        processing_jobs = trainings.get('results', [])
        print(f"Found {len(processing_jobs)} processing jobs")
        
        if not processing_jobs:
            print("No processing jobs found to monitor.")
            print("Returning empty status dictionary")
            return {}
        
        # Extract job IDs
        job_statuses = {job['id']: 'processing' for job in processing_jobs}
        print(f"\nInitial job statuses: {json.dumps(job_statuses, indent=2)}")
        
        # Monitor jobs until all are complete
        while any(status == 'processing' for status in job_statuses.values()):
            print(f"\nChecking job statuses at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            for job_id in list(job_statuses.keys()):
                if job_statuses[job_id] == 'processing':
                    try:
                        print(f"Checking status for job {job_id}...")
                        new_status = get_training_status(job_id)
                        print(f"Received status: {new_status}")
                        
                        if new_status != 'processing':
                            print(f"Job {job_id} completed with status: {new_status}")
                            job_statuses[job_id] = new_status
                    except Exception as e:
                        print(f"Error checking status for job {job_id}: {str(e)}")
                        print(f"Full error details: {traceback.format_exc()}")
            
            # If there are still processing jobs, wait before next check
            if any(status == 'processing' for status in job_statuses.values()):
                processing_count = sum(1 for status in job_statuses.values() if status == 'processing')
                print(f"Still waiting for {processing_count} job(s)...")
                print(f"Sleeping for {check_interval} seconds...")
                time.sleep(check_interval)
        
        print("\nAll jobs completed!")
        print(f"Final job statuses: {json.dumps(job_statuses, indent=2)}")
        return job_statuses
        
    except Exception as e:
        print(f"Error in monitor_training_jobs: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")
        raise


def get_model_details(owner: str, model_name: str) -> Dict[str, Any]:
    """
    Get details for a specific model from Replicate API.
    
    Args:
        owner (str): Username of the model owner
        model_name (str): Name of the model
    
    Returns:
        Dict[str, Any]: API response containing model details
    
    Raises:
        requests.exceptions.RequestException: If the API request fails
        KeyError: If REPLICATE_API_TOKEN environment variable is not set
    """
    
    # Get API token
    api_token = os.getenv('REPLICATE_API_KEY')
    if not api_token:
        raise KeyError("REPLICATE_API_KEY environment variable not found")
    
    # API endpoint
    url = f"https://api.replicate.com/v1/models/{owner}/{model_name}"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    
    try:
        # Make the GET request
        response = requests.get(
            url,
            headers=headers
        )
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting model details: {str(e)}")
        raise
    
def format_model_details(model: Dict[str, Any]) -> None:
    """
    Print formatted model details from the API response.
    
    Args:
        model (Dict[str, Any]): API response containing model details
    """
    print("\nModel Details:")
    print("-" * 80)
    
    # Basic model information
    print(f"Owner/Name: {model.get('owner', 'N/A')}/{model.get('name', 'N/A')}")
    print(f"Description: {model.get('description', 'N/A')}")
    print(f"Visibility: {model.get('visibility', 'N/A')}")
    print(f"Hardware: {model.get('hardware', 'N/A')}")
    print(f"URL: {model.get('url', 'N/A')}")
    
    # Created timestamp
    created_at = model.get('created_at', '')
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created_at = created_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            pass
    print(f"Created: {created_at}")
    
    # Latest version details
    latest_version = model.get('latest_version')
    if latest_version:
        print("\nLatest Version:")
        print(f"  ID: {latest_version.get('id', 'N/A')}")
        print(f"  Created: {latest_version.get('created_at', 'N/A')}")
        
        # Print available parameters if they exist
        openapi_schema = latest_version.get('openapi_schema')
        if openapi_schema and 'components' in openapi_schema:
            print("\nAvailable Parameters:")
            try:
                properties = openapi_schema['components']['schemas']['Input']['properties']
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'N/A')
                    param_description = param_info.get('description', 'No description available')
                    print(f"  {param_name}:")
                    print(f"    Type: {param_type}")
                    print(f"    Description: {param_description}")
            except KeyError:
                print("  Parameter information not available")
    
    print("-" * 80)
