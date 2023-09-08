
BUCKET_NAME=$1

if [ -z "$BUCKET_NAME" ]; then
    echo "Please provide bucket name."
    exit 1
fi

echo "BUCKET_NAME: $BUCKET_NAME"
python download_to_resource.py --bucket_name $BUCKET_NAME
python make_df_all.py --bucket_name $BUCKET_NAME