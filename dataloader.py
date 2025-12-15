"""
dataloader.py - Dataset loader for Nepali 20 Newsgroups (np20ng)
Handles loading from Hugging Face datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Any, Tuple, Optional # Added Optional
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
            self.dataset = load_dataset("Suyogyart/np20ng", split=split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting alternative loading method...")
            self.dataset = load_dataset("Suyogyart/np20ng")[split]
        
        # Get label names - Fix for the 'label' vs 'category' field
        # Check what field contains the labels
        if 'category' in self.dataset.features:
            self.label_field = 'category'
        elif 'label' in self.dataset.features:
            self.label_field = 'label'
        else:
            raise ValueError("Dataset must have 'category' or 'label' field")
        
        # Get label names
        if hasattr(self.dataset.features[self.label_field], 'names'):
            self.label_names = self.dataset.features[self.label_field].names
        else:
            # Extract unique labels
            unique_labels = sorted(set(self.dataset[self.label_field]))
            # Check if labels are already strings or integers
            if isinstance(unique_labels[0], str):
                self.label_names = unique_labels
            else:
                self.label_names = [f"class_{i}" for i in unique_labels]
        
        self.num_classes = len(self.label_names)
        
        # Create label to index mapping
        self.label2idx = {label: idx for idx, label in enumerate(self.label_names)}
        self.idx2label = {idx: label for idx, label in enumerate(self.label_names)}
        
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
    
    # MODIFICATION 1: Use Optional to indicate that None can be returned
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, Dict]]:
        """
        Get a single sample.
        
        Returns:
            input_ids, attention_mask, label, metadata, OR None if sample is invalid.
        """
        try:
            sample = self.dataset[idx]
            
            # Get text - dataset has 'content' field, not 'text'
            if 'content' in sample:
                text = sample['content']
            elif 'text' in sample:
                text = sample['text']
            else:
                # Fallback: combine heading and content if available
                text = sample.get('heading', '') + ' ' + sample.get('content', '')
            
            # Check for empty or excessively short text
            if not text or len(text.strip()) < 5:
                # Returning None here allows collate_fn to skip this sample
                return None 
            
            # Get label and convert to integer if it's a string
            label = sample[self.label_field]
            if isinstance(label, str):
                label = self.label2idx[label]
            
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
                'label_name': self.label_names[label] if isinstance(label, int) else self.idx2label.get(label, 'Unknown')
            }
            
            # Ensure label is a long tensor (required for CrossEntropyLoss)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return input_ids, attention_mask, label_tensor, metadata
        
        except Exception as e:
            # Log the error and return None to skip the sample
            # print(f"Error processing sample {idx}: {e}. Skipping.") 
            return None


def get_dataloaders(
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders
    """
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load the full dataset
    print("Loading full dataset from Hugging Face...")
    full_dataset = NepaliTextDataset(
        split='train',  # Only 'train' split exists
        model_name=model_name,
        max_length=max_length
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split_ratio * total_size)
    val_size = int(val_split_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplitting dataset:")
    print(f"  Total samples: {total_size}")
    print(f"  Train: {train_size} ({train_split_ratio*100:.1f}%)")
    print(f"  Val: {val_size} ({val_split_ratio*100:.1f}%)")
    print(f"  Test: {test_size} ({(1-train_split_ratio-val_split_ratio)*100:.1f}%)")
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
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
    
    # Get label names from the original dataset
    label_names = full_dataset.label_names
    
    print(f"\n{'='*60}")
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_loader.dataset)} samples)")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    print(f"  Max length: {max_length}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader, label_names


# MODIFICATION 2: Modified collate_fn to filter invalid samples
def collate_fn(batch):
    """
    Custom collate function for batching. 
    Filters out invalid samples (None) and only stacks valid data.
    """
    # Filter out any samples that were returned as None by __getitem__
    batch = [item for item in batch if item is not None]

    # If the entire batch was invalid or empty after filtering, return minimal empty tensors
    if not batch:
        # Return zero-sized tensors that won't crash the training loop,
        # but will be correctly skipped when checking batch size (e.g., if batch_size > 0)
        empty_ids = torch.empty((0, 0), dtype=torch.long)
        return empty_ids, empty_ids, torch.empty(0, dtype=torch.long), []

    # Stack the valid samples
    # item[0]=input_ids, item[1]=attention_masks, item[2]=label_tensor, item[3]=metadata
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    
    # The label is already a long tensor from __getitem__, but we re-cast for robustness
    labels = torch.stack([item[2] for item in batch]).long() 
    metadata = [item[3] for item in batch]
    
    return input_ids, attention_masks, labels, metadata


class DataStatistics:
    """Compute dataset statistics"""
    
    @staticmethod
    def compute_stats(dataset) -> Dict:
        """Compute comprehensive dataset statistics"""
        # Handle both Dataset and Subset
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            indices = dataset.indices
        else:
            base_dataset = dataset
            indices = range(len(dataset))
        
        stats = {
            'num_samples': len(indices),
            'num_classes': base_dataset.num_classes,
            'class_names': base_dataset.label_names,
            'class_distribution': {},
            'text_length_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'median': 0
            }
        }
        
        # Count class distribution
        # Note: Accessing base_dataset.dataset[i] might be slow inside compute_stats 
        # but is necessary to get the original label if random_split was used.
        labels = [base_dataset.dataset[i][base_dataset.label_field] for i in indices]
        for label in labels:
            # Convert string label to index if needed
            if isinstance(label, str):
                label_idx = base_dataset.label2idx.get(label, 0)
                label_name = base_dataset.label_names[label_idx]
            else:
                label_name = base_dataset.label_names[label]
            stats['class_distribution'][label_name] = stats['class_distribution'].get(label_name, 0) + 1
        
        # Compute text length statistics
        lengths = []
        for i in indices:
            sample = base_dataset.dataset[i]
            if 'content' in sample:
                text = sample['content']
            elif 'text' in sample:
                text = sample['text']
            else:
                text = sample.get('heading', '') + ' ' + sample.get('content', '')
            lengths.append(len(text))
        
        stats['text_length_stats']['min'] = min(lengths) if lengths else 0
        stats['text_length_stats']['max'] = max(lengths) if lengths else 0
        stats['text_length_stats']['mean'] = np.mean(lengths) if lengths else 0
        stats['text_length_stats']['median'] = np.median(lengths) if lengths else 0
        
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
            percentage = (count / stats['num_samples']) * 100 if stats['num_samples'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        print(f"\nText length statistics:")
        print(f"  Min: {stats['text_length_stats']['min']}")
        print(f"  Max: {stats['text_length_stats']['max']}")
        print(f"  Mean: {stats['text_length_stats']['mean']:.2f}")
        print(f"  Median: {stats['text_length_stats']['median']:.2f}")
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
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention masks shape: {attention_masks.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Sample text: {metadata[0]['text'][:100]}...")
        print(f"  Label name: {metadata[0]['label_name']}")
        
        # Compute statistics
        stats = DataStatistics.compute_stats(train_loader.dataset)
        DataStatistics.print_stats(stats)
        
        print("✓ Dataloader test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()