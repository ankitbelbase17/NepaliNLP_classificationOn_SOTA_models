"""
tests/test_model.py - Unit tests for Nepali text classification models
"""

import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model


class TestNepaliModels(unittest.TestCase):
    """Test all Nepali text classification models"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 128
        self.num_classes = 20
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_bert_forward(self):
        """Test BERT forward pass"""
        model = get_model('bert', self.num_classes)
        model.eval()
        
        with torch.no_grad():
            logits, aux = model(self.input_ids, self.attention_mask)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('attentions', aux)
        print("✓ BERT test passed")
    
    def test_nepbert_forward(self):
        """Test NepBERT forward pass"""
        model = get_model('nepbert', self.num_classes)
        model.eval()
        
        with torch.no_grad():
            logits, aux = model(self.input_ids, self.attention_mask)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        print("✓ NepBERT test passed")
    
    def test_t5_forward(self):
        """Test T5 forward pass"""
        model = get_model('t5', self.num_classes)
        model.eval()
        
        with torch.no_grad():
            logits, aux = model(self.input_ids, self.attention_mask)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        print("✓ T5 test passed")
    
    def test_electra_forward(self):
        """Test ELECTRA forward pass"""
        model = get_model('electra', self.num_classes)
        model.eval()
        
        with torch.no_grad():
            logits, aux = model(self.input_ids, self.attention_mask)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        print("✓ ELECTRA test passed")
    
    def test_gradient_flow(self):
        """Test gradient flow"""
        model = get_model('bert', self.num_classes)
        target = torch.randint(0, self.num_classes, (self.batch_size,))
        
        logits, _ = model(self.input_ids, self.attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_gradients)
        print("✓ Gradient flow test passed")
    
    def test_parameter_count(self):
        """Test parameter counting"""
        from utils import count_parameters
        
        model = get_model('bert', self.num_classes)
        params = count_parameters(model)
        
        self.assertGreater(params['total'], 0)
        self.assertEqual(params['total'], params['trainable'] + params['frozen'])
        print(f"✓ Parameter count test passed (Total: {params['total']:,})")

