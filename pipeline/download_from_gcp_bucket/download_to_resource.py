""" Please note your google credentaion should be set before this script.
    You can set it by:
    $ export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credential.json"
"""

import os
from google.cloud import storage
from argparse import ArgumentParser
from utils import mkdir_if_not_exists

# Replace with your project and bucket names
project_name = 'shine-mobiledoctor'

parser = ArgumentParser()
parser.add_argument("--bucket_name", type=str, help="Please write bucket name")
args = parser.parse_args()
bucket_name = args.bucket_name

base_destination = '../resource'
dir_destination = os.path.join(base_destination, bucket_name)

# Create a client object to access the Google Cloud Storage API
client = storage.Client(project=project_name)

# Get a reference to the bucket
bucket = client.get_bucket(bucket_name)

# Get a list of all the blobs (i.e., files) in the bucket
blobs = bucket.list_blobs()

# Create a directory to save the downloaded files
mkdir_if_not_exists(dir_destination)

# Download each file to the local directory
for blob in blobs:
    # Get the file name
    file_name = blob.name.split('/')[-1]

    if not file_name.startswith('df_'):
        continue

    print(f"Downloading {file_name} ...")

    # Download the file to the local directory
    file_fullpath = os.path.join(dir_destination, file_name)
    blob.download_to_filename(file_fullpath)
