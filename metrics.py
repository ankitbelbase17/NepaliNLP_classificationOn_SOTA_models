# ==============================================================================
# metrics.py - Same as CV project, works for text classification
# ==============================================================================

"""
metrics.py - Comprehensive evaluation metrics
(Reuses code from CV project - works for both image and text classification)
"""
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import json
import argparse
from tqdm import tqdm


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_logits: torch.Tensor,
    class_names: list
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = torch.softmax(y_logits, dim=1).numpy()
    
    num_classes = len(class_names)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # ROC-AUC
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            roc_auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
    except:
        roc_auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'kappa': kappa,
        'roc_auc': roc_auc
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str,
    normalize: bool = True
):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str = None
) -> str:
    """Generate classification report"""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(report)
    print("="*70)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Classification report saved to {save_path}")
    
    return report


def plot_per_class_metrics(
    metrics: Dict[str, float],
    class_names: list,
    save_path: str
):
    """Plot per-class precision, recall, F1"""
    precisions = [metrics.get(f'precision_{name}', 0) for name in class_names]
    recalls = [metrics.get(f'recall_{name}', 0) for name in class_names]
    f1_scores = [metrics.get(f'f1_{name}', 0) for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
    ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1_scores, width, label='F1 Score', color='#e74c3c')
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class metrics plot saved to {save_path}")


def save_metrics_json(metrics: Dict[str, float], save_path: str):
    """Save metrics to JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Metrics saved to {save_path}")


def generate_metrics_summary(
    metrics: Dict[str, float],
    class_names: list
):
    """Generate metrics summary"""
    import pandas as pd
    
    overall_metrics = {
        'Metric': [
            'Accuracy', 'Precision (Macro)', 'Precision (Weighted)',
            'Recall (Macro)', 'Recall (Weighted)', 'F1 (Macro)',
            'F1 (Weighted)', 'MCC', 'Cohen\'s Kappa', 'ROC-AUC'
        ],
        'Score': [
            metrics['accuracy'],
            metrics['precision_macro'],
            metrics['precision_weighted'],
            metrics['recall_macro'],
            metrics['recall_weighted'],
            metrics['f1_macro'],
            metrics['f1_weighted'],
            metrics['mcc'],
            metrics['kappa'],
            metrics['roc_auc']
        ]
    }
    
    df_overall = pd.DataFrame(overall_metrics)
    
    per_class_data = {
        'Class': class_names,
        'Precision': [metrics.get(f'precision_{name}', 0) for name in class_names],
        'Recall': [metrics.get(f'recall_{name}', 0) for name in class_names],
        'F1 Score': [metrics.get(f'f1_{name}', 0) for name in class_names]
    }
    
    df_per_class = pd.DataFrame(per_class_data)
    
    print("\n" + "="*70)
    print("Overall Metrics")
    print("="*70)
    print(df_overall.to_string(index=False))
    
    print("\n" + "="*70)
    print("Per-Class Metrics")
    print("="*70)
    print(df_per_class.to_string(index=False))
    print("="*70 + "\n")
    
    return df_overall, df_per_class


def compute_metrics_on_test_set(
    model: torch.nn.Module,
    test_loader,
    device: str,
    label_names: list,
    output_dir: str = None
):
    """
    Compute metrics on test set using model inference.
    
    Args:
        model: The model to use for inference (can be None if using pretrained without checkpoint)
        test_loader: DataLoader for test set (required)
        device: Device to use for inference
        label_names: List of class names
        output_dir: Directory to save results
    """
    all_labels = []
    all_logits = []
    all_predictions = []
    
    # Load test set and run inference
    print("Loading test set and running inference...")
    model.eval()
    
    with torch.no_grad():
        for input_ids, attention_masks, labels, metadata in tqdm(test_loader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            logits, aux_outputs = model(input_ids, attention_masks)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_labels, all_predictions, all_logits, label_names)
    
    # Generate outputs
    if output_dir is None:
        output_dir = './metrics_results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics summary
    generate_metrics_summary(metrics, label_names)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_predictions, label_names, cm_path, normalize=True)
    
    # Generate classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    generate_classification_report(all_labels, all_predictions, label_names, report_path)
    
    # Plot per-class metrics
    per_class_path = os.path.join(output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics, label_names, per_class_path)
    
    # Save metrics JSON
    metrics_json_path = os.path.join(output_dir, 'metrics.json')
    save_metrics_json(metrics, metrics_json_path)
    
    print("\n" + "="*70)
    print("Metrics Computation Complete")
    print("="*70)
    print(f"Test set size: {len(all_labels)}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return metrics


def main(args):
    """Main function for metrics computation"""
    from utils import load_checkpoint, get_device
    from model import get_model
    from dataloader import get_dataloaders, NepaliTextDataset
    
    device = get_device()
    
    print("\n" + "="*70)
    print("Metrics Computation")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # Load dataloaders (data_dir is required)
    print("Loading dataloaders...")
    _, _, test_loader, label_names = get_dataloaders(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # Load model
    print(f"\nLoading {args.model_name} model...")
    model = get_model(args.model_name, num_classes=len(label_names))
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model, device=device)
    else:
        print("✓ Using untrained model (pretrained weights only)")
    
    print("✓ Model loaded successfully")
    
    # Compute metrics
    metrics = compute_metrics_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        label_names=label_names,
        output_dir=args.output_dir
    )
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute metrics on test set'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['bert', 'bart', 'electra', 'reformer', 'mbart',
                'canine', 'nepbert', 't5', 'qwen2'],
        help='Model architecture'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional, uses pretrained weights if not provided)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for test set'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for data loading'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./metrics_results',
        help='Output directory for metrics and visualizations'
    )
    
    args = parser.parse_args()
    
    main(args)
