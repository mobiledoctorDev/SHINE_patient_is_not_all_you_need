
# 1. download shine file from GCP
cd download_from_gcp_bucket
./run.sh shine_v3_11_kt

# 2. add owid data
cd ../add_owid_data
./run.sh shine_v3_11_kt

# 3. postprocessing location data
cd ../loc_data_postprocessing
./run.sh shine_v3_11_kt
