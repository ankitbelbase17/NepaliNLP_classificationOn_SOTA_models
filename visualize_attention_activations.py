"""
visualize_attention.py - Visualize attention weights for Nepali text classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import List, Tuple

from model import get_model
from utils import load_checkpoint, get_device
from transformers import AutoTokenizer


def get_word_tokens(text: str) -> List[str]:
    """
    Split text into word tokens (Nepali words separated by spaces)
    
    Args:
        text: Input text
        
    Returns:
        List of word tokens
    """
    # Split by whitespace and filter empty strings
    words = [w.strip() for w in text.split() if w.strip()]
    return words


def get_attention_display_tokens(
    tokenizer_tokens: List[str],
    original_text: str,
    attention_mask
) -> List[str]:
    """
    Get display tokens from original text for visualization
    
    Args:
        tokenizer_tokens: Tokens from tokenizer
        original_text: Original input text
        attention_mask: Attention mask to determine actual length
        
    Returns:
        List of display tokens (words from original text)
    """
    # Get actual sequence length (excluding padding)
    actual_length = attention_mask.sum().item() if hasattr(attention_mask, 'sum') else len(tokenizer_tokens)
    
    # Get word tokens from original text
    word_tokens = get_word_tokens(original_text)
    
    # If we have word tokens, use them
    if word_tokens:
        # Truncate word tokens to match attention weights length
        return word_tokens[:min(len(word_tokens), int(actual_length))]
    else:
        # Fallback to tokenizer tokens
        return tokenizer_tokens[:int(actual_length)]



def visualize_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: str,
    layer_idx: int = -1,
    head_idx: int = 0
):
    """
    Visualize attention weights as heatmap
    
    Args:
        attention_weights: Attention weights (num_layers, num_heads, seq_len, seq_len)
        tokens: List of tokens
        save_path: Path to save visualization
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
    """
    # Get attention from specific layer and head
    attn = attention_weights[layer_idx][0, head_idx].cpu().detach().numpy()
    
    # Truncate tokens and attention if too long
    max_len = min(len(tokens), 50)
    tokens = tokens[:max_len]
    attn = attn[:max_len, :max_len]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot heatmap
    im = ax.imshow(attn, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Title
    ax.set_title(
        f'Attention Weights (Layer {layer_idx}, Head {head_idx})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.set_xlabel('Key Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Tokens', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention heatmap saved to {save_path}")


def visualize_all_heads(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: str,
    layer_idx: int = -1,
    max_heads: int = 12
):
    """Visualize attention from all heads"""
    attn = attention_weights[layer_idx][0].cpu().detach().numpy()
    num_heads = min(attn.shape[0], max_heads)
    
    # Truncate
    max_len = min(len(tokens), 30)
    tokens = tokens[:max_len]
    attn = attn[:, :max_len, :max_len]
    
    # Calculate grid
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        
        im = ax.imshow(attn[head_idx], cmap='YlOrRd', aspect='auto')
        ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'All Attention Heads (Layer {layer_idx})',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Multi-head visualization saved to {save_path}")


def visualize_token_attention_scores(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: str,
    layer_idx: int = -1,
    query_token_idx: int = 0
):
    """
    Visualize attention scores from a specific token to all others
    
    Args:
        attention_weights: Attention weights
        tokens: List of tokens
        save_path: Save path
        layer_idx: Layer index
        query_token_idx: Index of query token
    """
    # Average over all heads
    attn = attention_weights[layer_idx][0].cpu().detach().numpy()
    attn_avg = attn.mean(axis=0)  # Average over heads
    
    # Get attention from query token
    token_attn = attn_avg[query_token_idx]
    
    # Truncate
    max_len = min(len(tokens), 50)
    tokens = tokens[:max_len]
    token_attn = token_attn[:max_len]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    colors = ['#e74c3c' if i == query_token_idx else '#3498db' for i in range(len(tokens))]
    bars = ax.bar(range(len(tokens)), token_attn, color=colors)
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Attention Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tokens', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Attention Scores from "{tokens[query_token_idx]}" (Layer {layer_idx})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Token attention scores saved to {save_path}")


def visualize_attention_flow(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: str,
    num_layers: int = 4
):
    """Visualize how attention flows through layers"""
    # Take every few layers if too many
    layer_indices = np.linspace(0, len(attention_weights) - 1, num_layers, dtype=int)
    
    # Truncate tokens
    max_len = min(len(tokens), 30)
    tokens = tokens[:max_len]
    
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 5, 6))
    axes = [axes] if num_layers == 1 else axes
    
    for idx, layer_idx in enumerate(layer_indices):
        # Average over heads
        attn = attention_weights[layer_idx][0].cpu().detach().numpy()
        attn_avg = attn.mean(axis=0)[:max_len, :max_len]
        
        ax = axes[idx]
        im = ax.imshow(attn_avg, cmap='YlOrRd', aspect='auto')
        ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
        
        # Set y-axis ticks and labels
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_ylabel('Query Tokens', fontsize=10)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_xlabel('Key Tokens', fontsize=10)
    
    plt.suptitle(
        'Attention Flow Across Layers',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention flow visualization saved to {save_path}")


def main(args):
    # Get device
    device = get_device()
    
    print("\n" + "="*70)
    print("Attention and Activation Visualization")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    
    # Load tokenizer
    tokenizer_map = {
        'bert': 'bert-base-multilingual-cased',
        'nepbert': 'Rajan/NepaliBERT',
        'electra': 'google/electra-base-discriminator',
        't5': 'google/mt5-base',
        'qwen2': 'Qwen/Qwen2-0.5B'
    }
    
    tokenizer_name = tokenizer_map.get(args.model_name, 'bert-base-multilingual-cased')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Load dataset and dataloader
    from dataloader import NepaliTextDataset, get_dataloaders
    
    if args.text:
        # Mode 1: Use provided text
        print(f"\nMode: Using provided text")
        print("="*70 + "\n")
        
        # Tokenize text
        encoding = tokenizer(
            args.text,
            max_length=args.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get tokens for visualization
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Remove padding tokens
        actual_length = attention_mask[0].sum().item()
        tokens = tokens[:actual_length]
        
        # Get label names
        temp_dataset = NepaliTextDataset(split='train', model_name=args.model_name)
        num_classes = temp_dataset.num_classes
        label_names = temp_dataset.label_names
        
    else:
        # Mode 2: Use first batch from dataloader
        print(f"Mode: Using first batch from dataloader")
        print("="*70 + "\n")
        
        print("Loading dataloader...")
        train_loader, _, _, label_names = get_dataloaders(
            model_name=args.model_name,
            batch_size=1,
            max_length=args.max_length,
            num_workers=0
        )
        
        num_classes = len(label_names)
        
        # Get first batch
        batch_iter = iter(train_loader)
        input_ids, attention_mask, labels, metadata = next(batch_iter)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get tokens for visualization
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Remove padding tokens
        actual_length = attention_mask[0].sum().item()
        tokens = tokens[:actual_length]
        
        # Extract text from metadata
        # metadata is a list of dicts, so get first item
        text = metadata[0]['text'] if isinstance(metadata, list) else metadata['text']
        true_label = label_names[labels[0].item()]
        
        print(f"Text: {text}")
        print(f"True label: {true_label}")
        print(f"Sequence length: {actual_length}")
        print()
    
    # Create model
    model = get_model(args.model_name, num_classes=num_classes)
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model, device=device)
        print("✓ Model loaded successfully\n")
    
    model.eval()
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        logits, aux_outputs = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    print(f"Predicted class: {label_names[pred_class]}")
    print(f"Confidence: {pred_prob:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize if attention weights available
    if aux_outputs.get('attentions') is not None:
        print("\nGenerating attention visualizations...")
        
        # Get display tokens (original text words instead of subword tokens)
        # For text mode, use the provided text
        if args.text:
            display_text = args.text
        else:
            display_text = text
        
        display_tokens = get_attention_display_tokens(tokens, display_text, attention_mask)
        
        # Single head heatmap
        visualize_attention_heatmap(
            aux_outputs['attentions'],
            display_tokens,
            os.path.join(args.output_dir, 'attention_heatmap.png'),
            layer_idx=args.layer_idx,
            head_idx=args.head_idx
        )
        
        # All heads
        visualize_all_heads(
            aux_outputs['attentions'],
            display_tokens,
            os.path.join(args.output_dir, 'attention_all_heads.png'),
            layer_idx=args.layer_idx
        )
        
        # Token attention scores
        visualize_token_attention_scores(
            aux_outputs['attentions'],
            display_tokens,
            os.path.join(args.output_dir, 'token_attention_scores.png'),
            layer_idx=args.layer_idx,
            query_token_idx=0
        )
        
        # Attention flow
        visualize_attention_flow(
            aux_outputs['attentions'],
            display_tokens,
            os.path.join(args.output_dir, 'attention_flow.png')
        )
        
        print(f"\n✓ All visualizations saved to: {args.output_dir}")
    else:
        print("\n⚠ No attention weights available for this model")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize attention weights for Nepali text classification'
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
        '--text',
        type=str,
        default=None,
        help='Nepali text to analyze (optional, uses first batch from dataloader if not provided)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint (optional)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='attention_visualizations',
        help='Output directory'
    )
    parser.add_argument(
        '--layer_idx',
        type=int,
        default=-1,
        help='Layer index (-1 for last)'
    )
    parser.add_argument(
        '--head_idx',
        type=int,
        default=0,
        help='Attention head index'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    
    args = parser.parse_args()
    main(args)