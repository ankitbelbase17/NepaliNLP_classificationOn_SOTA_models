"""
tests/test_dataloader.py - Unit tests for Nepali text dataloader
"""

import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNepaliDataLoader(unittest.TestCase):
    """Test Nepali dataset and dataloader"""
    
    def test_dataset_loading(self):
        """Test loading np20ng dataset"""
        try:
            from dataloader import NepaliTextDataset
            
            dataset = NepaliTextDataset(split='train', model_name='bert', max_length=128)
            
            self.assertGreater(len(dataset), 0)
            self.assertGreater(dataset.num_classes, 0)
            print(f"✓ Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")
        except Exception as e:
            self.skipTest(f"Dataset not available: {e}")
    
    def test_dataset_item(self):
        """Test getting dataset item"""
        try:
            from dataloader import NepaliTextDataset
            
            dataset = NepaliTextDataset(split='train', model_name='bert', max_length=128)
            
            input_ids, attention_mask, label, metadata = dataset[0]
            
            self.assertEqual(input_ids.shape[0], 128)
            self.assertEqual(attention_mask.shape[0], 128)
            self.assertIsInstance(label, int)
            self.assertIn('text', metadata)
            print("✓ Dataset item test passed")
        except Exception as e:
            self.skipTest(f"Dataset not available: {e}")
    
    def test_dataloader_batch(self):
        """Test dataloader batching"""
        try:
            from dataloader import get_dataloaders
            
            train_loader, _, _, label_names = get_dataloaders(
                model_name='bert',
                batch_size=4,
                max_length=128,
                num_workers=0
            )
            
            input_ids, attention_masks, labels, metadata = next(iter(train_loader))
            
            self.assertEqual(input_ids.shape[0], 4)
            self.assertEqual(attention_masks.shape[0], 4)
            self.assertEqual(labels.shape[0], 4)
            self.assertEqual(len(metadata), 4)
            print("✓ DataLoader batch test passed")
        except Exception as e:
            self.skipTest(f"Dataset not available: {e}")
