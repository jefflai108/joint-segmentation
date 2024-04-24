import torch
import unittest

from src.utils import *

class TestAccuracyMetrics(unittest.TestCase):
    def test_accuracy_with_mask_consistency1(self):
        # Generate random logits and targets for testing
        logits = torch.randn(10, 10)  # example size, can vary
        targets = torch.randint(0, 2, (10, 10)).float()  # binary targets
        targets[:, -5:] = PAD_TOKEN

        # Call both functions
        _, _, _, boundary_accuracy = calculate_boundary_metrics(logits, targets, ignore_index=PAD_TOKEN)
        simple_accuracy = calculate_accuracy(logits, targets, ignore_index=PAD_TOKEN)

        # Assert that both accuracies are approximately equal
        print(boundary_accuracy) 
        print(simple_accuracy)
        self.assertAlmostEqual(boundary_accuracy, simple_accuracy, places=5,
                               msg="The accuracies from both functions should be approximately equal.")

    def test_accuracy_with_mask_consistency2(self):
        # Generate random logits and targets for testing
        logits = torch.randn(4, 4)  # example size, can vary
        targets = torch.randint(0, 2, (4, 4)).float()  # binary targets
        targets[:, -2:] = PAD_TOKEN

        # Call both functions
        f1, _, _, boundary_accuracy = calculate_boundary_metrics(logits, targets, ignore_index=PAD_TOKEN)
        simple_accuracy = calculate_accuracy(logits, targets, ignore_index=PAD_TOKEN)

        print(f1)
        # Assert that both accuracies are approximately equal
        print(boundary_accuracy) 
        print(simple_accuracy)
        self.assertAlmostEqual(boundary_accuracy, simple_accuracy, places=5,
                               msg="The accuracies from both functions should be approximately equal.")

if __name__ == '__main__':
    unittest.main()

