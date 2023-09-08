import os
import urllib.request

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--bucket_name", type=str, help="Please write bucket name")
args = parser.parse_args()
bucket_name = args.bucket_name
assert bucket_name is not None

# Replace with the URL of the file you want to download
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

# Replace with the name you want to save the file as
file_name = 'owid-covid-data.csv'
save_dir = f'../resource/{args.bucket_name}'
dst_filepath = os.path.join(save_dir, file_name)

# Download the file and save it to the local machine
urllib.request.urlretrieve(url, dst_filepath)