# ==============================================================================
# scripts/metrics.sh - Metrics evaluation
# ==============================================================================

#!/bin/bash

MODEL_NAME=${1:-bert}
CHECKPOINT=${2:-./experiments/checkpoints/${MODEL_NAME}_best.pth}
OUTPUT_DIR=${3:-./metrics_results}

echo "========================================"
echo "Evaluating Metrics"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "========================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint not found"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

python -c "
import torch
import json
import os
from model import get_model
from dataloader import get_dataloaders
from utils import load_checkpoint, get_device
from metrics import (
    calculate_metrics, plot_confusion_matrix,
    generate_classification_report, plot_per_class_metrics,
    save_metrics_json, generate_metrics_summary
)
from tqdm import tqdm

device = get_device()
model_name = '${MODEL_NAME}'
checkpoint_path = '${CHECKPOINT}'
output_dir = '${OUTPUT_DIR}'

print('Loading data...')
_, _, test_loader, label_names = get_dataloaders(
    model_name=model_name,
    batch_size=32,
    num_workers=4
)

print(f'Loading {model_name} model...')
model = get_model(model_name, num_classes=len(label_names))
model = model.to(device)
load_checkpoint(checkpoint_path, model, device=device)
model.eval()

print('Running inference...')
all_predictions = []
all_labels = []
all_logits = []

with torch.no_grad():
    for input_ids, attention_masks, labels, _ in tqdm(test_loader):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        logits, _ = model(input_ids, attention_masks)
        _, predicted = torch.max(logits, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_logits.append(logits.cpu())

all_logits = torch.cat(all_logits, dim=0)

print('Calculating metrics...')
metrics = calculate_metrics(all_labels, all_predictions, all_logits, label_names)

print('Generating visualizations...')
plot_confusion_matrix(
    all_labels, all_predictions, label_names,
    os.path.join(output_dir, 'confusion_matrix.png'),
    normalize=True
)

generate_classification_report(
    all_labels, all_predictions, label_names,
    os.path.join(output_dir, 'classification_report.txt')
)

plot_per_class_metrics(
    metrics, label_names,
    os.path.join(output_dir, 'per_class_metrics.png')
)

save_metrics_json(metrics, os.path.join(output_dir, 'metrics.json'))

df_overall, df_per_class = generate_metrics_summary(metrics, label_names)

print(f'\n✓ All metrics saved to: {output_dir}')
"

echo "✓ Metrics evaluation completed"

