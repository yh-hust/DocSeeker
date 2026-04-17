DATASETS_TO_PROCESS="dude SlideVQA mpdocvqa"
#DATASETS_TO_PROCESS="LongDocURL"
# done: LongDocURL mpdocvqa dude
# to do: mmlongbench-doc DocBench SlideVQA mmlongbench-doc
# These variables remain constant for all runs.
DEVICE="0,1,2,3,4,5,6,7"
SUBSET="val"
EXPERIMENT_NAME=qwen2_5vl-flash_attn_2_data_expand_all_data_correct_mpdocvqa
checkpoint_id=checkpoint-5000
MODEL_PATH="/home/ma-user/work/dataset/dataset_yh/docseeker_weight/${EXPERIMENT_NAME}/dude_mpdocvqa_yh_dynamic_random_pixels_1024_256/${checkpoint_id}"
EVAL_MODEL='gemini-2.5-flash'
# torchrun settings (assuming 8 GPUs based on the DEVICE variable)
export NPROC_PER_NODE=8
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yunwu.ai/v1}"

# --- Main Loop ---
# Loop over each dataset name defined in the DATASETS_TO_PROCESS variable.
for dataset_name in ${DATASETS_TO_PROCESS}; do
    file_to_check="${dataset_name}/output/${EXPERIMENT_NAME}_${checkpoint_id}/${SUBSET}/${dataset_name}.json"
    if [ -f "${file_to_check}" ]; then
        echo "推理已完成！"
    else
        # 如果文件仍不存在，打印提示信息并等待一段时间后重试
        echo "进行推理..."        
    fi

    while [ ! -f "${file_to_check}" ]; do
        
        echo "目标文件不存在: ${file_to_check}"
        echo "即将执行推理命令..."
        echo "======================================================"
        echo "🚀 Starting process for dataset: ${dataset_name}"
        echo "======================================================"

        # 1. Run VLM Inference
        # Arguments for the inference script, updated with the current dataset name.
        args="
            --model_path ${MODEL_PATH} \
            --dataset_name ${dataset_name} \
            --save_dir ${EXPERIMENT_NAME} \
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
        # 执行您定义的命令
        echo "---"
        echo "Executing command: ${cmd_str}"
        echo "---"
        eval ${cmd_str}
        
        # 检查命令执行后文件是否已经生成
    done

    # Define the output path for the calculation scripts.
    # 2. Run LLM Calculation
    output_path="${dataset_name}/${EXPERIMENT_NAME}/${SUBSET}"
    file_to_check="${dataset_name}/output/${EXPERIMENT_NAME}_${checkpoint_id}/${SUBSET}/${dataset_name}_${EVAL_MODEL}_metric.json"
    if [ -f "${file_to_check}" ]; then
        echo "抽取的文件存在！"
    else
        # 如果文件仍不存在，打印提示信息并等待一段时间后重试
        echo "抽取未完成，将进行答案抽取" 
        echo "---"
        echo "Running llm_cal.py on output path: ${output_path}"
        echo "---"
        python llm_cal.py --output_path ${output_path}  --model_name ${EVAL_MODEL}
    fi

    # 3. Run Metric Calculation
    echo "---"
    echo "Running metric_cal.py on output path: ${output_path}"
    echo "---"
    python metric_cal.py --output_path ${output_path}  --model_name ${EVAL_MODEL}

    echo "✅ Finished processing for dataset: ${dataset_name}"
    echo ""
done
