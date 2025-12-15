"""
inference.py - Inference script for Nepali text classification
"""

import torch
import torch.nn.functional as F
import argparse
import os
import json
from tqdm import tqdm

from model import get_model
from dataloader import get_dataloaders, NepaliTextDataset
from utils import load_checkpoint, get_device
from metrics import calculate_metrics
from transformers import AutoTokenizer


def predict_single_text(
    model: torch.nn.Module,
    text: str,
    tokenizer,
    label_names: list,
    device: str,
    max_length: int = 512,
    top_k: int = 5
) -> dict:
    """Predict class for a single text"""
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        logits, aux_outputs = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(label_names)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': label_names[idx.item()],
            'class_idx': idx.item(),
            'probability': prob.item()
        })
    
    return {
        'text': text,
        'predictions': predictions,
        'top_prediction': predictions[0]
    }


def batch_inference(
    model: torch.nn.Module,
    dataloader,
    device: str,
    label_names: list,
    save_results: bool = True,
    output_dir: str = None
) -> dict:
    """Run inference on batch of texts"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    all_texts = []
    
    print("Running batch inference...")
    with torch.no_grad():
        for input_ids, attention_masks, labels, metadata in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            logits, aux_outputs = model(input_ids, attention_masks)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu())
            all_texts.extend([m['text'] for m in metadata])
    
    # Concatenate all logits
    all_logits = torch.cat(all_logits, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_labels, all_predictions, all_logits, label_names
    )
    
    # Prepare results
    results = {
        'num_samples': len(all_predictions),
        'metrics': metrics,
        'predictions': []
    }
    
    # Add individual predictions
    for i in range(len(all_predictions)):
        pred_probs = F.softmax(all_logits[i], dim=0)
        top_prob, top_idx = torch.max(pred_probs, 0)
        
        results['predictions'].append({
            'text': all_texts[i][:200] + '...' if len(all_texts[i]) > 200 else all_texts[i],
            'true_label': label_names[all_labels[i]],
            'true_label_idx': int(all_labels[i]),
            'predicted_label': label_names[all_predictions[i]],
            'predicted_label_idx': int(all_predictions[i]),
            'confidence': float(top_prob.item()),
            'correct': all_labels[i] == all_predictions[i]
        })
    
    # Save results
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'inference_results.json')
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Inference Summary")
    print("="*60)
    print(f"Total samples: {results['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print("="*60)
    
    return results


def main(args):
    device = get_device()
    
    if args.mode == 'batch':
        # Load dataset
        from dataloader import get_dataloaders
        _, _, test_loader, label_names = get_dataloaders(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers
        )
    else:
        # For single text, need tokenizer and label names
        tokenizer_map = {
            'bert': 'bert-base-multilingual-cased',
            'nepbert': 'Rajan/NepaliBERT',
            't5': 'google/mt5-base',
            'qwen2': 'Qwen/Qwen2-0.5B'
        }
        tokenizer_name = tokenizer_map.get(args.model_name, 'bert-base-multilingual-cased')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Load label names from file
        if args.label_names_file and os.path.exists(args.label_names_file):
            with open(args.label_names_file, 'r', encoding='utf-8') as f:
                label_names = json.load(f)
        else:
            # Load from dataset to get labels
            temp_dataset = NepaliTextDataset(split='train', model_name=args.model_name)
            label_names = temp_dataset.label_names
    
    # Create model
    print(f"\nLoading {args.model_name} model...")
    model = get_model(args.model_name, num_classes=len(label_names))
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")
    
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Run inference
    if args.mode == 'single':
        result = predict_single_text(
            model, args.text, tokenizer,
            label_names, device, args.max_length, args.top_k
        )
        
        print("\n" + "="*60)
        print(f"Text: {args.text}")
        print("="*60)
        print("\nPredictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"{i}. {pred['class']}: {pred['probability']:.4f}")
        print("="*60)
        
        # Save result
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 'prediction.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Result saved to: {output_path}")
    
    else:
        results = batch_inference(
            model, test_loader, device, label_names,
            save_results=True, output_dir=args.output_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['bert', 'bart', 'electra', 'reformer', 'mbart',
                               'canine', 'nepbert', 't5', 'qwen2'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'batch'],
                        help='Inference mode')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to classify (for single mode)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for batch inference')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory')
    parser.add_argument('--label_names_file', type=str, default=None,
                        help='Path to label names JSON')
    
    args = parser.parse_args()
    
    if args.mode == 'single' and not args.text:
        parser.error("--text required for single mode")
    
    main(args)