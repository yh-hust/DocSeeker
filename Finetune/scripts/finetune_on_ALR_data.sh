#!/bin/bash
NNODES=${MA_NUM_HOSTS:-1}
NODE_RANK=${VC_TASK_INDEX:-0}
NPROC_PER_NODE=${MA_NUM_GPUS:-8}
if [ -n "$vc_worker_hosts" ]; then
    MASTER_ADDR=$(echo "$vc_worker_hosts" | cut -d ',' -f 1)
else
    MASTER_ADDR="127.0.0.1"
fi
MASTER_PORT="6060"

export nnodes=${NNODES}
export node_rank=${NODE_RANK}
export master_addr=${MASTER_ADDR}
export master_port=${MASTER_PORT}

DEEPSPEED_CONFIG="scripts/zero3.json"
MODEL_PATH="your/path/to/Qwen2.5-VL-7B-Instruct"
ENTRY_FILE="qwenvl/train/train_qwen.py"

# 数据集与实验命名
DATASETS=""
RUN_NAME=""
ATTN_IMPLEMENTATION="flash_attention_2"
SAVE_DIR_NAME=""

# 输出目录拼接 (处理了原代码中 output_dir 嵌套的问题)
BASE_OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"} # 如果外部没传 OUTPUT_DIR，默认存在当前目录下的 outputs
FINAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}/${SAVE_DIR_NAME}"

# 创建输出目录
mkdir -p "${FINAL_OUTPUT_DIR}"

# ==========================================
# 3. 超参数设置
# ==========================================
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
EVAL_BATCH_SIZE=$(( BATCH_SIZE * 2 ))
LEARNING_RATE="2e-6"


DISTRIBUTED_ARGS=(
    "--nproc_per_node" "${NPROC_PER_NODE}"
    "--nnodes" "${NNODES}"
    "--node_rank" "${NODE_RANK}"
    "--master_addr" "${MASTER_ADDR}"
    "--master_port" "${MASTER_PORT}"
)

TRAIN_ARGS=(
    "--deepspeed" "${DEEPSPEED_CONFIG}"
    "--model_name_or_path" "${MODEL_PATH}"
    "--dataset_use" "${DATASETS}"
    "--data_flatten" "True"
    "--tune_mm_vision" "True"
    "--tune_mm_mlp" "True"
    "--tune_mm_llm" "True"
    "--bf16"
    "--output_dir" "${FINAL_OUTPUT_DIR}"
    "--num_train_epochs" "6.0"
    "--per_device_train_batch_size" "${BATCH_SIZE}"
    "--per_device_eval_batch_size" "${EVAL_BATCH_SIZE}"
    "--gradient_accumulation_steps" "${GRAD_ACCUM_STEPS}"
    "--max_pixels" "802816"
    "--min_pixels" "12544"
    "--eval_strategy" "no"
    "--save_strategy" "epoch"
    "--learning_rate" "${LEARNING_RATE}"
    "--mm_projector_lr" "1e-5"
    "--vision_tower_lr" "1e-6"
    "--weight_decay" "0"
    "--warmup_ratio" "0.1"
    "--max_grad_norm" "1"
    "--lr_scheduler_type" "cosine"
    "--logging_steps" "1"
    "--model_max_length" "17408"
    "--gradient_checkpointing" "True"
    "--dataloader_num_workers" "4"
    "--run_name" "${RUN_NAME}"
    "--report_to" "none"
    "--attn_implementation" "${ATTN_IMPLEMENTATION}"
    "--use_egra" "True"
    "--low_max_pixels" "200704"
)

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
mkdir -p "logs"
LOG_FILE="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=================================================="
echo "Starting Qwen2.5-VL Training..."
echo "Time          : ${TIMESTAMP}"
echo "Nodes         : ${NNODES} (Current Rank: ${NODE_RANK})"
echo "GPUs/Node     : ${NPROC_PER_NODE}"
echo "Output Dir    : ${FINAL_OUTPUT_DIR}"
echo "Log File      : ${LOG_FILE}"
echo "=================================================="

torchrun "${DISTRIBUTED_ARGS[@]}" "${ENTRY_FILE}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
