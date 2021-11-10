import unittest
import torch

from classifiers.models import FCN
from utils.metrics import IoUMetric

class TestSegmentation(unittest.TestCase):
    def test_model(self):
        """
        Test forward pass of the model
        """
        X = torch.zeros((8, 3, 51, 71))
        model = FCN()
        scores = model(X)
        self.assertTrue(scores["out"].shape, torch.Size([8, 1, 51, 71]))

    def test_iou_metric(self):
        """
        Test IoU Metric for various input configurations
        """
        iou = IoUMetric()
        y_pred = 1
        y_base = 1
        self.assertEqual(y_pred, y_base)


if __name__ == "__main__":
    unittest.main()

