#!/bin/bash

# =============================================================================
# Predefined variables and configurations
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MIXTURE_SCRIPT="${SCRIPT_DIR}/solution/get_mixture.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/toolkit/training/examples/train.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/toolkit/evaluation/examples/eval.sh"

CONFIG_FILE="${SCRIPT_DIR}/entry.env"
source ${CONFIG_FILE}

# Function to add parameters to a command
add_param() {
    local -n cmd_ref=$1
    local flag=$2
    local value=$3
    if [ -n "$value" ]; then
        cmd_ref+=" ${flag} ${value}"
    fi
}

# Function for printing log messages
log() {
    if [[ $# -eq 0 ]]; then
        echo
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    fi
}

# =============================================================================
# Mixture Part
# =============================================================================
log "Starting Mixture Process..."

mixture_cmd="python ${MIXTURE_SCRIPT}"
log "${mixture_cmd}"
eval "${mixture_cmd}"

if [ $? -ne 0 ]; then
    echo "Error: Mixture failed with exit code $?."
    exit 1
fi

# =============================================================================
# Training Part
# =============================================================================
log
log "Starting Training Process..."

train_cmd="bash ${TRAIN_SCRIPT}"
add_param train_cmd "--model" "baichuan-inc/Baichuan2-7B-Base"
add_param train_cmd "--int8" "${INT8}"
add_param train_cmd "--dtype" "${COMPUTE_TYPE}"
add_param train_cmd "--data" "${SCRIPT_DIR}/output/sft_data/mixture.jsonl"
add_param train_cmd "--batch_size" "${MICRO_BATCH_SIZE}"
add_param train_cmd "--pack" "${SFT_PACKING}"
add_param train_cmd "--output" "${SCRIPT_DIR}/output/lora_model"
add_param train_cmd "--ds_stage" "${DS_ZERO_STAGE}"
add_param train_cmd "--ds_offload" "${DS_OFFLOAD}"
add_param train_cmd "--lr" "${LEARNING_RATE}"
case "${LEARNING_RATE}" in
    1e-3|1e-4|1e-5)
        add_param train_cmd "--lr" "${LEARNING_RATE}"
        ;;
    *)
        echo "LEARNING_RATE is not one of 1e-3, 1e-4, or 1e-5. Aborting."
        exit 1
        ;;
esac
log "${train_cmd}"
eval "${train_cmd}"

if [ $? -ne 0 ]; then
    echo "Error: Training failed with exit code $?."
    exit 1
fi

# =============================================================================
# Evaluation Part
# =============================================================================
log
log "Starting Evaluation Process..."

eval_cmd="bash ${EVAL_SCRIPT}"
add_param eval_cmd "--mode" "${EVAL_MODE}"
add_param eval_cmd "--model" "baichuan-inc/Baichuan2-7B-Base"
add_param eval_cmd "--lora" "${SCRIPT_DIR}/output/lora_model"
add_param eval_cmd "--int8" "${INT8}"
add_param eval_cmd "--dtype" "${COMPUTE_TYPE}"
add_param eval_cmd "--data" "${SCRIPT_DIR}/toolkit/evaluation/data"
add_param eval_cmd "--output" "${SCRIPT_DIR}/output/evals"
add_param eval_cmd "--seqlen" "1024"
log "${eval_cmd}"
eval "${eval_cmd}"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed with exit code $?."
    exit 1
fi

# =============================================================================
# Finalization
# =============================================================================
log
log "Script execution completed."
