
BUCKET_NAME=$1

if [ -z "$BUCKET_NAME" ]; then
    echo "Please provide bucket name."
    exit 1
fi

echo "BUCKET_NAME: $BUCKET_NAME"

# 1. raw gps date -> individual daily location data into shine table (long_std, lati_std, n, path_sum, ...)
# this could be run by daily basis. Like,
# python post_processor_loc_shine.py --n_thread 20

# 2. Save shine table (shine2_pp_loc_data) into csv file
# When you want. This will go to ./tmp_output because this is a temporary result before the training. like,
#python pp_loc_to_csv.py
echo "************************************"
echo Working bucket: $BUCKET_NAME
echo "************************************"
echo "\n"

# 2.5. copy df_all to ./tmp_output (df_data_v3_1/df_all_v3_12_added_owid.csv)
echo "************************************"
echo "Copying files ../output/$BUCKET_NAME/df_all_added_owid.csv"
echo "  -> ./tmp_output/$BUCKET_NAME/df_all_added_owid.csv"
echo "************************************"

mkdir ./tmp_output/$BUCKET_NAME/
cp ../output/$BUCKET_NAME/df_all_added_owid.csv ./tmp_output/$BUCKET_NAME/df_all_added_owid.csv
echo "\n"

# 3. normalize the data using Quantile Transformation (long_std_norm, lati_std_norm, n_norm, path_sum_norm, ...)
echo "************************************"
echo "Generating mask_and_normalized location data."
echo "************************************"
python mask_and_normalize.py --bucket_name $BUCKET_NAME
echo "\n"

# 3. muti-days of location data to
echo "************************************"
echo "Merging patient and location data into ./tmp_output/$BUCKET_NAME/df_all_added_owid_loc.csv"
echo "************************************"
python merge_patient_and_loc_norm.py --bucket_name $BUCKET_NAME
echo "\n"

# 4. Copy normalized data and function to ../output
echo "************************************"
echo "Copying files ./tmp_output/$BUCKET_NAME/df_all_added_owid_loc.csv -> ../output/"
echo "Copying files ./tmp_output/$BUCKET_NAME/pp_params -> ../output/$BUCKET_NAME/pp_params/"
echo "************************************"
cp ./tmp_output/$BUCKET_NAME/df_all_added_owid_loc.csv ../output/$BUCKET_NAME
cp ./tmp_output/$BUCKET_NAME/gps_full_2023-04-13_norm.csv ../output/$BUCKET_NAME
cp ./tmp_output/$BUCKET_NAME/bts_full_2023-04-13_norm.csv ../output/$BUCKET_NAME
cp -r ./tmp_output/$BUCKET_NAME/pp_params/ ../output/$BUCKET_NAME/pp_params/
echo "\n"
