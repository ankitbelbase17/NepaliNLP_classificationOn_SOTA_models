"""
utils.py - Utility functions for Nepali text classification
(Same structure as CV project, adapted for text)
"""

import torch
import torch.nn as nn
import os
import json
import yaml
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_wandb(config: Dict[str, Any], project_name: str = "nepali_text_classification"):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=project_name,
        config=config,
        name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=[config['model_name'], 'nepali', 'np20ng']
    )
    print(f"✓ WandB initialized: {wandb.run.name}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    iteration: int,
    best_val_acc: float,
    save_dir: str,
    model_name: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    import glob
    os.makedirs(save_dir, exist_ok=True)

    # Delete previous checkpoints
    for ckpt in glob.glob(os.path.join(save_dir, f'{model_name}_latest.pth')):
        os.remove(ckpt)
    for ckpt in glob.glob(os.path.join(save_dir, f'{model_name}_iter_*.pth')):
        os.remove(ckpt)
    if is_best:
        for ckpt in glob.glob(os.path.join(save_dir, f'{model_name}_best.pth')):
            os.remove(ckpt)

    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'model_name': model_name
    }

    # Save latest
    latest_path = os.path.join(save_dir, f'{model_name}_latest.pth')
    torch.save(checkpoint, latest_path)

    # Save iteration checkpoint
    iter_path = os.path.join(save_dir, f'{model_name}_iter_{iteration}.pth')
    torch.save(checkpoint, iter_path)

    # Save best
    if is_best:
        best_path = os.path.join(save_dir, f'{model_name}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved with val_acc: {best_val_acc:.4f}")

    print(f"✓ Checkpoint saved at iteration {iteration}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return {'epoch': 0, 'iteration': 0, 'best_val_acc': 0.0}
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']}")
    print(f"  Best val acc: {checkpoint.get('best_val_acc', 0.0):.4f}")
    
    return checkpoint


def log_to_wandb(metrics: Dict[str, float], step: int, images: Optional[Dict] = None):
    """Log metrics to WandB"""
    log_dict = {**metrics, 'step': step}
    
    if images:
        wandb_images = {}
        for key, img_data in images.items():
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.cpu().numpy()
            wandb_images[key] = wandb.Image(img_data)
        log_dict.update(wandb_images)
    
    wandb.log(log_dict, step=step)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str
):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', linewidth=2, color='#3498db')
    ax1.plot(val_losses, label='Val Loss', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Acc', linewidth=2, color='#2ecc71')
    ax2.plot(val_accs, label='Val Acc', linewidth=2, color='#f39c12')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {save_path}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


def get_device() -> torch.device:
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device


def create_experiment_dir(base_dir: str, model_name: str) -> str:
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'{model_name}_{timestamp}')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    print(f"✓ Experiment directory: {exp_dir}")
    
    return exp_dir


