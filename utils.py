from google.cloud import storage
from google.cloud.storage import Blob
from google.oauth2 import service_account
import datetime
import os

import yaml
from typing import Dict, Optional


class GCSBucketManager:
    """
    A class to manage Google Cloud Storage bucket operations.

    This class provides methods for common GCS operations including:
    - Connecting to a bucket
    - Uploading files
    - Downloading files
    - Deleting files
    - Managing public access
    - Generating signed URLs
    """

    def __init__(self, bucket_name: str, credentials_path: str):
        """
        Initialize the GCS Bucket Manager.

        Args:
            bucket_name (str): Name of the GCS bucket
            credentials_path (str): Path to service account JSON key file
        """
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.credentials = None
        self.client = None
        self.bucket = None
        self._connect()

    def _connect(self) -> None:
        """
        Establish connection to the GCS bucket.

        Raises:
            Exception: If connection fails
        """
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.client = storage.Client(credentials=self.credentials)
            self.bucket = self.client.bucket(self.bucket_name)
        except Exception as e:
            raise Exception(f"Failed to connect to GCS bucket: {str(e)}")

    def upload_file(self, source_path: str, destination_blob_name: str = None) -> str:
        """
        Upload a file to the bucket.

        Args:
            source_path (str): Local path to the file to upload
            destination_blob_name (str, optional): Desired path/name in GCS.
                                                 If None, uses the source filename

        Returns:
            str: GCS path of the uploaded file

        Raises:
            FileNotFoundError: If source file doesn't exist
            Exception: If upload fails
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        try:
            if destination_blob_name is None:
                destination_blob_name = os.path.basename(source_path)

            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_path)

            return destination_blob_name
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    def download_file(self, blob_name: str, destination_path: str) -> str:
        """
        Download a file from the bucket.

        Args:
            blob_name (str): Path/name of the file in GCS
            destination_path (str): Local path where file should be downloaded

        Returns:
            str: Local path of the downloaded file

        Raises:
            Exception: If download fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(destination_path)
            return destination_path
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")

    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a file from the bucket.

        Args:
            blob_name (str): Path/name of the file in GCS

        Returns:
            bool: True if deletion was successful

        Raises:
            Exception: If deletion fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            raise Exception(f"Failed to delete file: {str(e)}")

    def make_blob_public(self, blob_name: str) -> str:
        """
        Make a blob publicly accessible and return its public URL.

        Args:
            blob_name (str): Path/name of the file in GCS

        Returns:
            str: Public URL of the blob

        Raises:
            Exception: If making blob public fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            raise Exception(f"Failed to make blob public: {str(e)}")

    def get_public_url(self, blob_name: str) -> str:
        """
        Get the public URL for a blob. Note that this method only returns the URL -
        it does not check if the blob is actually publicly accessible.

        Args:
            blob_name (str): Path/name of the file in GCS

        Returns:
            str: Public URL of the blob

        Raises:
            Exception: If getting the URL fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            return blob.public_url
        except Exception as e:
            raise Exception(f"Failed to get public URL: {str(e)}")

    def check_public_access(self, blob_name: str) -> bool:
        """
        Check if a blob is publicly accessible.

        Args:
            blob_name (str): Path/name of the file in GCS

        Returns:
            bool: True if the blob is publicly accessible, False otherwise

        Raises:
            Exception: If checking access fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            policy = blob.acl.get_entry("allUsers")
            return policy is not None and policy.role == "READER"
        except Exception as e:
            raise Exception(f"Failed to check public access: {str(e)}")

    def generate_signed_url(self, blob_name: str, expiration: int = 3600) -> str:
        """
        Generate a signed URL for temporary access to a blob.

        Args:
            blob_name (str): Path/name of the file in GCS
            expiration (int): Time in seconds until URL expires (default: 1 hour)

        Returns:
            str: Signed URL for temporary access

        Raises:
            Exception: If URL generation fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(seconds=expiration),
                method="GET",
            )
            return signed_url
        except Exception as e:
            raise Exception(f"Failed to generate signed URL: {str(e)}")

    def list_files(self, prefix: str = None) -> list:
        """
        List all files in the bucket, optionally filtered by prefix.

        Args:
            prefix (str, optional): Filter results to files starting with this prefix

        Returns:
            list: List of blob names
        """
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            raise Exception(f"Failed to list files: {str(e)}")

    def set_cors_policy(
        self, origins: list = None, methods: list = None, max_age_seconds: int = 3600
    ) -> None:
        """
        Set CORS policy for the bucket.

        Args:
            origins (list): List of allowed origins (default: ["*"])
            methods (list): List of allowed methods (default: ["GET", "HEAD"])
            max_age_seconds (int): Max age for CORS policy (default: 1 hour)

        Raises:
            Exception: If setting CORS policy fails
        """
        try:
            if origins is None:
                origins = ["*"]
            if methods is None:
                methods = ["GET", "HEAD"]

            self.bucket.cors = [
                {
                    "origin": origins,
                    "method": methods,
                    "responseHeader": ["Content-Type"],
                    "maxAgeSeconds": max_age_seconds,
                }
            ]
            self.bucket.patch()
        except Exception as e:
            raise Exception(f"Failed to set CORS policy: {str(e)}")


class ModelConfig:
    """
    A class to handle model configuration from YAML file
    """

    def __init__(self, yaml_path: str):
        """
        Initialize ModelConfig with path to YAML file

        Args:
            yaml_path (str): Path to the YAML configuration file
        """
        self.yaml_path = yaml_path
        self.models: Dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        Load model configurations from YAML file

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is malformed
        """
        try:
            with open(self.yaml_path, "r") as file:
                config = yaml.safe_load(file)
                self.models = config.get("models", {})
        except FileNotFoundError:
            # If file doesn't exist, initialize with empty models dict
            self.models = {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {str(e)}")

    def _save_config(self) -> None:
        """
        Save current configuration to YAML file

        Raises:
            IOError: If unable to write to file
        """
        try:
            config = {"models": self.models}
            with open(self.yaml_path, "w") as file:
                yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        except IOError as e:
            raise IOError(f"Error saving configuration to file: {str(e)}")

    def add_or_update_model(self, model_name: str, model_url: str) -> bool:
        """
        Add a new model or update an existing model's URL

        Args:
            model_name (str): Name of the model
            model_url (str): URL of the model

        Returns:
            bool: True if operation was successful

        Raises:
            ValueError: If model_name or model_url is empty
            IOError: If unable to save to file
        """
        # Validate inputs
        if not model_name or not model_name.strip():
            raise ValueError("Model name cannot be empty")
        if not model_url or not model_url.strip():
            raise ValueError("Model URL cannot be empty")

        try:
            # Update the models dictionary
            self.models[model_name] = model_url

            # Save changes to file
            self._save_config()

            return True
        except Exception as e:
            raise IOError(f"Failed to update model configuration: {str(e)}")

    def get_model_url(self, model_name: str) -> Optional[str]:
        """
        Get URL for a specific model

        Args:
            model_name (str): Name of the model

        Returns:
            Optional[str]: URL of the model if found, None otherwise
        """
        return self.models.get(model_name)

    def list_models(self) -> list:
        """
        Get list of all available model names

        Returns:
            list: List of model names
        """
        return list(self.models.keys())

    def get_all_models(self) -> Dict[str, str]:
        """
        Get dictionary of all models and their URLs

        Returns:
            Dict[str, str]: Dictionary with model names as keys and URLs as values
        """
        return self.models


# Example usage:
# # Initialize config
# model_config = ModelConfig('models.yaml')

# try:
# 	# Add or update a model
# 	model_config.add_or_update_model(
# 		"gpt-4-turbo",
# 		"https://api.openai.com/v1/models/gpt-4-turbo"
# 	)
# 	print("Model added/updated successfully!")

# 	# Verify the update
# 	print("\nUpdated model list:")
# 	for model, url in model_config.get_all_models().items():
# 		print(f"{model}: {url}")

# except ValueError as e:
# 	print(f"Validation error: {str(e)}")
# except IOError as e:
# 	print(f"File operation error: {str(e)}")
# except Exception as e:
# 	print(f"Unexpected error: {str(e)}")
