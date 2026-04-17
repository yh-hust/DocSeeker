# Available models:
# - Qwen2.5-VL-7B-Instruct
# - Custom fine-tuned checkpoints
# Python entry script:
# - vlm_inference_on_mpdoc.py
# Supported datasets (only these 5 are allowed):
# - mmlongbench_doc
# - LongDocURL
# - dude
# - SlideVQA
# - mpdocvqa
# Supported subsets:
# - test
# - val
# Subset availability by dataset:
# - mpdocvqa: test, val
# - dude: test, val
# - SlideVQA: test, val
# - mmlongbench_doc: test only
# - LongDocURL: test only

DATASETS_TO_PROCESS="mmlongbench_doc LongDocURL dude SlideVQA mpdocvqa"
# DATASETS_TO_PROCESS="LongDocURL"
# Completed: LongDocURL mpdocvqa dude SlideVQA mmlongbench_doc
# TODO: mmlongbench_doc

DEVICE="0,1,2,3,4,5,6,7"
SUBSET="test"
EXPERIMENT_NAME="" 
checkpoint_id=""
MODEL_PATH=""
EVAL_MODEL=""
# torchrun settings (assuming 8 GPUs based on the DEVICE variable)
export NPROC_PER_NODE=8
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

export OPENAI_API_KEY="${OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"

# --- Main Loop ---
# Loop over each dataset name defined in the DATASETS_TO_PROCESS variable.
for dataset_name in ${DATASETS_TO_PROCESS}; do
    file_to_check="${dataset_name}/output/${EXPERIMENT_NAME}_${checkpoint_id}/${SUBSET}/${dataset_name}.json"
    if [ -f "${file_to_check}" ]; then
        echo "Inference has already been completed."
    else
        # If the file still does not exist, print a message and retry after running inference
        echo "Running inference..."        
    fi

    while [ ! -f "${file_to_check}" ]; do
        
        echo "Target file does not exist: ${file_to_check}"
        echo "Preparing to run the inference command..."
        echo "======================================================"
        echo "🚀 Starting process for dataset: ${dataset_name}"
        echo "======================================================"

        # 1. Run VLM Inference
        # Arguments for the inference script, updated with the current dataset name.
        args="
            --model_path ${MODEL_PATH} \
            --dataset_name ${dataset_name} \
            --save_dir "${dataset_name}/output/${EXPERIMENT_NAME}_${checkpoint_id}/${SUBSET}" \
            --subset ${SUBSET} \
            --use_page_named True
            "

        # The full command to execute for inference.
        cmd_str="CUDA_VISIBLE_DEVICES=${DEVICE} torchrun --nproc_per_node=${NPROC_PER_NODE} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            vlm_inference_on_mpdoc.py ${args}
            "
        
        echo "---"
        echo "Executing inference command for ${dataset_name}..."
        echo "${cmd_str}"
        echo "---"
        # Execute the command defined above
        echo "---"
        echo "Executing command: ${cmd_str}"
        echo "---"
        eval ${cmd_str}
        
        # Check whether the file has been generated after the command finishes
    done

    # Define the output path for the calculation scripts.
    # 2. Run LLM Calculation
    output_path="${dataset_name}/output/${EXPERIMENT_NAME}_${checkpoint_id}/${SUBSET}"
    file_to_check="${dataset_name}/output/${EXPERIMENT_NAME}_${checkpoint_id}/${SUBSET}/${dataset_name}_${EVAL_MODEL}_metric.json"
    if [ -f "${file_to_check}" ]; then
        echo "The extracted file already exists."
    else
        # If the file still does not exist, print a message and run answer extraction
        echo "Extraction has not been completed. Running answer extraction now." 
        echo "---"
        echo "Running llm_cal.py on output path: ${output_path}"
        echo "---"
        python llm_cal.py --output_path ${output_path}  --model_name ${EVAL_MODEL}
    fi

    # 3. Run Metric Calculation
    echo "---"
    echo "Running metric_cal.py on output path: ${output_path}"
    echo "---"
    python metric_cal.py --output_path ${output_path} --model_name ${EVAL_MODEL}

    echo "✅ Finished processing for dataset: ${dataset_name}"
    echo ""
done
