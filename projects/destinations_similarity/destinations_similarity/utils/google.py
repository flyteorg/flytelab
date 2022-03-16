"""Auxiliary functions to use with Google Cloud."""

import os

from google.cloud import storage


class GoogleUtils():
    """Provides utilities for Google Cloud operations.

    The class uses the google-cloud module, that expects a credentials file.
    For ways to define the file, see the official documentation of the library:
    https://googleapis.dev/python/google-api-core/latest/auth.html.
    """

    def __init__(self):
        """Configure the object."""
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'bi-data-science')
        self.init_clients()

    def init_clients(self):
        """Init Google Cloud Plataform client services."""
        self.storage_client = storage.Client(project=self.project_id)

    def download_blob(self, bucket_name, source_path, destination_path):
        """Download a blob from a specified bucket.

        Args:
            bucket_name (str): Bucket name.
            source_path (str): Remote path of the blob on the bucket.
            destination_path (str): Desired local path for blob.
        """
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_path)
        blob.download_to_filename(destination_path)
        print(
            f"File gs://{bucket_name}/{source_path} downloaded to "
            f"{destination_path}."
        )

    def upload_blob(self, bucket_name, source_path, destination_path):
        """Upload a blob to a specified bucket.

        Args:
            bucket_name (str): Bucket name.
            source_path (str): Local path of the blob.
            destination_path (str): Desired remote path on the bucket.
        """
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_filename(source_path)
        print(
            f"File {source_path} uploaded to gs://{bucket_name}/"
            f"{destination_path}."
        )
