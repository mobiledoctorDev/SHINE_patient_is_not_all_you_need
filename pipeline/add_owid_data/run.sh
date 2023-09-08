
BUCKET_NAME=$1

if [ -z "$BUCKET_NAME" ]; then
    echo "Please provide bucket name."
    exit 1
fi

echo "BUCKET_NAME: $BUCKET_NAME"

python download_owid.py --bucket_name $BUCKET_NAME
python merge_owid_and_patient.py --bucket_name $BUCKET_NAME