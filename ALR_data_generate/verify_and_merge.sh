OUTPUT_PATH="/path/to/your/generated_data_dir"   # Path to the directory containing generated ALR results
MODEL_NAME="API_MODEL_NAME"                # Name of the model used for verification (e.g., gpt-4o)

# 1. Run secondary verification
python verification.py \
    --file_path "${OUTPUT_PATH}" \
    --model_name "${MODEL_NAME}"

# 2. Merge the verified results
python merge.py \
    --folder_path "${OUTPUT_PATH}"
