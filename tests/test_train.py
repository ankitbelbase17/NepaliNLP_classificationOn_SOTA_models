"""
tests/test_train.py - Unit tests for training pipeline
"""

import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model


class TestTraining(unittest.TestCase):
    """Test training pipeline"""
    
    def test_loss_calculation(self):
        """Test loss calculation"""
        criterion = torch.nn.CrossEntropyLoss()
        
        logits = torch.randn(4, 20)
        labels = torch.randint(0, 20, (4,))
        
        loss = criterion(logits, labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        print("✓ Loss calculation test passed")
    
    def test_optimizer_step(self):
        """Test optimizer step"""
        model = get_model('bert', 20)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Get initial params
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        target = torch.randint(0, 20, (2,))
        
        logits, _ = model(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        changed = False
        for p_initial, p_current in zip(initial_params, model.parameters()):
            if not torch.equal(p_initial, p_current):
                changed = True
                break
        
        self.assertTrue(changed)
        print("✓ Optimizer step test passed")
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save/load"""
        from utils import save_checkpoint, load_checkpoint
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            model = get_model('bert', 20)
            optimizer = torch.optim.AdamW(model.parameters())
            
            save_checkpoint(
                model, optimizer, None,
                epoch=2, iteration=50, best_val_acc=0.75,
                save_dir=temp_dir, model_name='test_model'
            )
            
            new_model = get_model('bert', 20)
            new_optimizer = torch.optim.AdamW(new_model.parameters())
            
            checkpoint_path = os.path.join(temp_dir, 'test_model_latest.pth')
            checkpoint = load_checkpoint(checkpoint_path, new_model, new_optimizer)
            
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['iteration'], 50)
            print("✓ Checkpoint save/load test passed")
            
        finally:
            shutil.rmtree(temp_dir)
