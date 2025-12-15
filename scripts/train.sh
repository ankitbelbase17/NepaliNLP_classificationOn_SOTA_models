# ==============================================================================
# scripts/train.sh - Training script for Nepali text classification
# Usage: bash scripts/train.sh <model_name>
# ==============================================================================

MODEL_NAME=${1:-bert}
BATCH_SIZE=${2:-16}
EPOCHS=${3:-10}
LR=${4:-2e-5}
MAX_LENGTH=${5:-512}
OUTPUT_DIR=${6:-./experiments}

echo "========================================"
echo "Training Nepali Text Classifier"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LR}"
echo "Max Length: ${MAX_LENGTH}"
echo "========================================"

# Validate model name
valid_models=("bert" "bart" "electra" "reformer" "mbart" "canine" "nepbert" "t5" "qwen2")
if [[ ! " ${valid_models[@]} " =~ " ${MODEL_NAME} " ]]; then
    echo "Error: Invalid model '${MODEL_NAME}'"
    echo "Valid models: ${valid_models[@]}"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training
python train.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --max_length ${MAX_LENGTH} \
    --resume

echo "✓ Training completed for ${MODEL_NAME}"
