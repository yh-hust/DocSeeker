timestamp=$(date "+%Y%m%d_%H%M%S")
INPUT_DATA="/path/to/your/raw_data.jsonl"                # Path to the processed multi-page VQA .jsonl dataset
IMAGE_DIR="/path/to/your/images_directory"               # Base directory where document images are stored
SAVE_BASE_DIR="/path/to/save/output_data"                # Root directory to save the generated ALR results
MODEL_NAME="API_MODEL_NAME"                              # Name of the teacher model (e.g., gemini-2.5-flash)
START_IDX="YOUR_START_IDX"                               # Start index for data slicing (inclusive)
END_IDX="YOUR_END_IDX"                                   # End index for data slicing (exclusive)
OUTPUT_NAME="mpdoc_expand_data_${START_IDX}_${END_IDX}"  # Folder name for this specific data chunk
OUTPUT_PATH="${SAVE_BASE_DIR}/${OUTPUT_NAME}"            # Full output path for this specific chunk
mkdir -p "${SAVE_BASE_DIR}"
python generator.py \
    --original_path "${INPUT_DATA}" \
    --save_path "${OUTPUT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --image_base_path "${IMAGE_DIR}" \
    --bg "${START_IDX}" \
    --ed "${END_IDX}"
