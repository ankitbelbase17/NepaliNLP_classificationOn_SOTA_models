
MODEL_NAME=${1:-bert}
CHECKPOINT=${2:-./experiments/checkpoints/${MODEL_NAME}_best.pth}
MODE=${3:-batch}
TEXT=${4:-""}
OUTPUT_DIR=${5:-./inference_results}

echo "========================================"
echo "Running Inference"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Mode: ${MODE}"
echo "========================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint '${CHECKPOINT}' not found"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

if [ "${MODE}" == "batch" ]; then
    python inference.py \
        --model_name ${MODEL_NAME} \
        --checkpoint ${CHECKPOINT} \
        --mode batch \
        --batch_size 32 \
        --output_dir ${OUTPUT_DIR}
elif [ "${MODE}" == "single" ]; then
    if [ -z "${TEXT}" ]; then
        echo "Error: Text required for single mode"
        exit 1
    fi
    
    python inference.py \
        --model_name ${MODEL_NAME} \
        --checkpoint ${CHECKPOINT} \
        --mode single \
        --text "${TEXT}" \
        --output_dir ${OUTPUT_DIR}
else
    echo "Error: Invalid mode '${MODE}'"
    exit 1
fi

echo "✓ Inference completed"
