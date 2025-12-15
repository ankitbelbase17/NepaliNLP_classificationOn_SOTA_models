"""
dataloader.py - Dataset loader for Nepali 20 Newsgroups (np20ng)
Handles loading from Hugging Face datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Tuple, Dict, List
import numpy as np


class NepaliTextDataset(Dataset):
    """Dataset for Nepali text classification"""
    
    def __init__(
        self,
        split: str = 'train',
        model_name: str = 'bert',
        max_length: int = 512,
        use_cache: bool = True
    ):
        """
        Initialize Nepali text dataset
        
        Args:
            split: 'train', 'validation', or 'test'
            model_name: Model name to determine tokenizer
            max_length: Maximum sequence length
            use_cache: Whether to cache dataset
        """
        self.split = split
        self.max_length = max_length
        
        # Load dataset from Hugging Face
        print(f"Loading np20ng dataset (split: {split})...")
        try:
            self.dataset = load_dataset("Suyogyart/np20ng", split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting alternative loading method...")
            self.dataset = load_dataset("Suyogyart/np20ng", trust_remote_code=True)[split]
        
        # Get label names
        if hasattr(self.dataset.features['label'], 'names'):
            self.label_names = self.dataset.features['label'].names
        else:
            # Extract unique labels
            unique_labels = sorted(set(self.dataset['label']))
            self.label_names = [f"class_{i}" for i in unique_labels]
        
        self.num_classes = len(self.label_names)
        
        # Load tokenizer based on model
        self.tokenizer = self._get_tokenizer(model_name)
        
        print(f"✓ Loaded {len(self.dataset)} samples")
        print(f"✓ Number of classes: {self.num_classes}")
        print(f"✓ Classes: {self.label_names}")
        
    def _get_tokenizer(self, model_name: str):
        """Get appropriate tokenizer for model"""
        tokenizer_map = {
            'bert': 'bert-base-multilingual-cased',
            'bart': 'facebook/mbart-large-cc25',
            'electra': 'google/electra-base-discriminator',
            'reformer': 'google/reformer-enwik8',
            'mbart': 'facebook/mbart-large-50',
            'canine': 'google/canine-s',
            'nepbert': 'Rajan/NepaliBERT',
            't5': 'google/mt5-base',
            'qwen2': 'Qwen/Qwen2-0.5B'
        }
        
        model_id = tokenizer_map.get(model_name.lower(), 'bert-base-multilingual-cased')
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except:
            print(f"Warning: Could not load tokenizer for {model_id}, using BERT")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """
        Get a single sample
        
        Returns:
            input_ids: Token IDs
            attention_mask: Attention mask
            label: Class label
            metadata: Additional information
        """
        sample = self.dataset[idx]
        
        text = sample['text']
        label = sample['label']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        metadata = {
            'idx': idx,
            'text': text,
            'label_name': self.label_names[label] if label < len(self.label_names) else f"class_{label}"
        }
        
        return input_ids, attention_mask, label, metadata


def get_dataloaders(
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        model_name: Model name for tokenizer selection
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader, label_names
    """
    
    # Create datasets
    train_dataset = NepaliTextDataset(
        split='train',
        model_name=model_name,
        max_length=max_length
    )
    
    # Check if validation split exists
    try:
        val_dataset = NepaliTextDataset(
            split='validation',
            model_name=model_name,
            max_length=max_length
        )
    except:
        # If no validation split, use test
        print("No validation split found, using test split for validation")
        val_dataset = NepaliTextDataset(
            split='test',
            model_name=model_name,
            max_length=max_length
        )
    
    try:
        test_dataset = NepaliTextDataset(
            split='test',
            model_name=model_name,
            max_length=max_length
        )
    except:
        print("No test split found, creating from train")
        # Split train into train and test
        from torch.utils.data import random_split
        train_size = int(0.9 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    label_names = train_dataset.label_names if hasattr(train_dataset, 'label_names') else train_dataset.dataset.label_names
    
    print(f"\n{'='*60}")
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_loader.dataset)} samples)")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    print(f"  Max length: {max_length}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader, label_names


def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    
    return input_ids, attention_masks, labels, metadata


class DataStatistics:
    """Compute dataset statistics"""
    
    @staticmethod
    def compute_stats(dataset: NepaliTextDataset) -> Dict:
        """Compute comprehensive dataset statistics"""
        stats = {
            'num_samples': len(dataset),
            'num_classes': dataset.num_classes,
            'class_names': dataset.label_names,
            'class_distribution': {},
            'text_length_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'median': 0
            }
        }
        
        # Count class distribution
        labels = [dataset.dataset[i]['label'] for i in range(len(dataset))]
        for label in labels:
            label_name = dataset.label_names[label]
            stats['class_distribution'][label_name] = stats['class_distribution'].get(label_name, 0) + 1
        
        # Compute text length statistics
        lengths = [len(dataset.dataset[i]['text']) for i in range(len(dataset))]
        stats['text_length_stats']['min'] = min(lengths)
        stats['text_length_stats']['max'] = max(lengths)
        stats['text_length_stats']['mean'] = np.mean(lengths)
        stats['text_length_stats']['median'] = np.median(lengths)
        
        return stats
    
    @staticmethod
    def print_stats(stats: Dict):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Total samples: {stats['num_samples']}")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"\nClass distribution:")
        for class_name, count in stats['class_distribution'].items():
            percentage = (count / stats['num_samples']) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        print(f"\nText length statistics:")
        print(f"  Min: {stats['text_length_stats']['min']}")
        print(f"  Max: {stats['text_length_stats']['max']}")
        print(f"  Mean: {stats['text_length_stats']['mean']:.2f}")
        print(f"  Median: {stats['text_length_stats']['median']:.2f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test dataloader
    print("Testing dataloader...")
    
    try:
        train_loader, val_loader, test_loader, label_names = get_dataloaders(
            model_name='bert',
            batch_size=4,
            max_length=128,
            num_workers=0
        )
        
        # Test batch
        input_ids, attention_masks, labels, metadata = next(iter(train_loader))
        print(f"\nBatch test:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention masks shape: {attention_masks.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Sample text: {metadata[0]['text'][:100]}...")
        print(f"  Label name: {metadata[0]['label_name']}")
        
        # Compute statistics
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'dataset'):
            train_dataset = train_dataset.dataset
        
        stats = DataStatistics.compute_stats(train_dataset)
        DataStatistics.print_stats(stats)
        
        print("✓ Dataloader test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()