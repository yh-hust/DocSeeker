timestamp=$(date "+%Y%m%d_%H%M%S")
INPUT_DATA="data/raw_data/train_dude.jsonl"
IMAGE_DIR="/home/ma-user/work/dataset/dataset_yh/docseeker_traindata/ablation/images"
SAVE_BASE_DIR="data/output_data"
MODEL_NAME="gemini-2.5-flash"
START_IDX=10
END_IDX=15
OUTPUT_NAME="mpdoc_expand_data_${START_IDX}_${END_IDX}"
OUTPUT_PATH="${SAVE_BASE_DIR}/${OUTPUT_NAME}"
mkdir -p "${SAVE_BASE_DIR}"
python generator.py \
    --original_path "${INPUT_DATA}" \
    --save_path "${OUTPUT_PATH}" \
    --model "${MODEL_NAME}" \
    --image_base_path "${IMAGE_DIR}" \
    --bg "${START_IDX}" \
    --ed "${END_IDX}"