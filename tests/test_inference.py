"""
tests/test_inference.py - Unit tests for inference pipeline
"""

import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model


class TestInference(unittest.TestCase):
    """Test inference pipeline"""
    
    def test_single_prediction(self):
        """Test single text prediction"""
        model = get_model('bert', 20)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 128))
        attention_mask = torch.ones(1, 128)
        
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
        
        self.assertEqual(logits.shape, (1, 20))
        print("✓ Single prediction test passed")
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        model = get_model('bert', 20)
        model.eval()
        
        batch = torch.randint(0, 1000, (8, 128))
        attention_mask = torch.ones(8, 128)
        
        with torch.no_grad():
            logits, _ = model(batch, attention_mask)
        
        self.assertEqual(logits.shape, (8, 20))
        print("✓ Batch prediction test passed")
    
    def test_top_k_predictions(self):
        """Test top-k predictions"""
        import torch.nn.functional as F
        
        model = get_model('bert', 20)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 128))
        attention_mask = torch.ones(1, 128)
        
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs[0], k=5)
        
        self.assertEqual(len(top_probs), 5)
        self.assertTrue(torch.all(top_probs[:-1] >= top_probs[1:]))
        print("✓ Top-k predictions test passed")
    
    def test_confidence_scores(self):
        """Test confidence scores"""
        import torch.nn.functional as F
        
        model = get_model('bert', 20)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (4, 128))
        attention_mask = torch.ones(4, 128)
        
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        self.assertTrue(torch.all(confidences >= 0))
        self.assertTrue(torch.all(confidences <= 1))
        print("✓ Confidence scores test passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("Running Nepali Text Classification Tests")
    print("="*70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestNepaliModels))
    suite.addTests(loader.loadTestsFromTestCase(TestNepaliDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestInference))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
